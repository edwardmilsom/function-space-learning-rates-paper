
import torch
from torch import nn

import math
from warnings import warn

# AKA Phi. The gradient of this is called "phigrad" in the code.
class RandomLinComb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        rand = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        return (x * rand).sum() / math.sqrt(x.numel())

# Function-space Layerwise Learning Rates. Code is written for clarity, not efficiency.
class WrapperUpdateNormaliser(nn.Module):
    def _get_named_parameters(self):
        if self._named_parameters is not None:
            return iter((name, param) for name, param in self._named_parameters)
        else:
            return iter((name, param) for name, param in self.model.named_parameters())
    def _get_parameters(self):
        if self._named_parameters is not None:
            return iter(param for name, param in self._named_parameters)
        else:
            return iter(param for param in self.model.parameters())
    # Should we make the user change beta based on how often they call this, or should we make them provide the number of iterations between calls?
    def __init__(self, model, optimiser, outerlr=0.1, masses=None,
                 scheduler=None, approx_type="kronecker",
                 beta=0.999, named_parameters=None, joint_parameters=None,
                 fix_nans=True,
                 dataloader=None,
                 eps=0.):
        super().__init__()
        self.joint_parameters = joint_parameters
        self.eps = eps
        self.fix_nans = fix_nans
        self.dataloader = dataloader

        self._named_parameters = None
        self.num_params = len(named_parameters) if named_parameters is not None else len(list(model.parameters()))
        if named_parameters is None:
            warn("we will use the all the parameters based on the [model] object! this may not be what you want (e.g. if using LoRA)")
        else:
            self._named_parameters = named_parameters

        self.model = model
        self.optimiser = optimiser

        self.set_masses(masses)

        if scheduler is None:
            self.scheduler = None
        else:
            raise NotImplementedError("Scheduler not implemented yet")

        device = next(self._get_parameters()).device
        dtype = next(self._get_parameters()).dtype
        self.device = device

        #Initialise stuff
        self.step = 1
        self.beta = beta
        self.approx_type = approx_type
        self.phi_module = RandomLinComb().to(next(model.parameters()).device)
        self.outerlr = outerlr

        ### Initialise buffers containing the EMAs for the \Delta_\ell F estimators. Different methods (iid, full_cov, kronecker) require different setups.
        if approx_type == "iid":
            self.iid_var_EMAs = []
            for name, param in self._get_named_parameters():
                name = name.replace(".", "_")
                self.register_buffer(f"{name}_iid_var_EMA", torch.zeros(1, device=param.device, dtype=param.dtype))
                self.iid_var_EMAs.append(getattr(self, f"{name}_iid_var_EMA"))

        elif approx_type == "full_cov":
            self.full_cov_var_EMAs = []
            for name, param in self._get_named_parameters():
                name = name.replace(".", "_")
                self.register_buffer(f"{name}_full_cov_var_EMA", torch.zeros(1, device=param.device, dtype=param.dtype))
                self.full_cov_var_EMAs.append(getattr(self, f"{name}_full_cov_var_EMA"))

        elif approx_type == "kronecker":
            self.kron_var_num_EMAs = [] # There are d "numerator" scalars to track for each d-dimensional parameter tensor (e.g. a conv filter has 4 dimensions)
            self.kron_var_denom_EMAs = [] # There is only one "denominator" scalar to track for each d-dimensional parameter tensor
            for name, param in self._get_named_parameters():
                name = name.replace(".", "_") # Replace dots with underscores to make it a valid attribute name
                self.register_buffer(f"{name}_kron_var_num_EMA", torch.zeros(max(param.ndim, 1), device=param.device, dtype=param.dtype)) # Scalar parameters have ndim=0, but we want to treat them as 1D, so we take the max of 1 and the actual ndim
                self.register_buffer(f"{name}_kron_var_denom_EMA", torch.zeros(1, device=param.device, dtype=param.dtype))
                self.kron_var_num_EMAs.append(getattr(self, f"{name}_kron_var_num_EMA"))
                self.kron_var_denom_EMAs.append(getattr(self, f"{name}_kron_var_denom_EMA"))

        else:
            raise ValueError(f"Invalid approx_type: {approx_type}. Must be one of 'iid', 'full_cov', 'kronecker', 'iid_vec_kron_tensor'.")

        ### Initialise buffer containing the EMA for the \Delta F estimator
        self.aggregate_var_EMA = torch.zeros(1, device=device, dtype=dtype)

    # masses is a dictionary of the form {parameter_name: mass}, assumed to contain all parameters in the model.
    # Controls contribution of each parameter to total function update, i.e. \Delta F AFTER normalisation.
    def set_masses(self, masses):

        num_params = sum(1 for _ in self._get_parameters())

        # If no masses are given, initialise them all equally. Suboptimal for almost all cases.
        if masses is None or masses == {}:
            named_parameters = dict(self._get_named_parameters())
            self.masses = [1/num_params for _ in range(num_params)]
            print("Warning: No masses provided. All parameters will be updated with equal mass adding to 1, which is suboptimal for almost all cases. To surpress this warning, go into the code and comment it out ;)")

        else:
            # If masses is not a dict, throw an error
            if not isinstance(masses, dict):
                raise ValueError("Provided masses must be a dictionary specifying the mass for all the named parameters. Use the same names as in model.named_parameters().")

            # Check that all the keys in masses are in the model
            named_parameters = dict(self._get_named_parameters())
            for name in masses:
                if name not in named_parameters:
                    raise ValueError(f"Parameter {name} not found in model")

            # Construct the masses list, spreading the remaining mass equally among the parameters not specified in masses
            self.masses = [masses[name] for name in named_parameters]
            assert len(self.masses) == num_params

    # Store the current weights so \Delta W can be computed.
    # Note we use self.weight_updates to save memory later.
    # We will in-place update self.weight_updates to become the difference between the new weights and the old weights in the method "update_lrs()".
    def save_weights(self):
        with torch.no_grad():
            self.weight_updates = [param.data.clone() for param in self._get_parameters()]

    # Compute width and depth invariant learning rates and set them in the optimiser.
    # 1. Compute \Delta W using saved weights and current weights, undo the last update on the weights, then divide by the learning rate to get the lr=1 update.
    # 2. Compute Phi, and do a backwards pass to get phigrads (dPhi/dW)
    # 3. Compute \Delta_\ell F for each layer using phigrads
    # 4. Set new learning rates
    # 5. Apply the normalised update (probably not necessary, but feels like a good idea)
    def update_lrs(self, batchinputs, return_delta_ell_fs=False, modify_lrs=True, kwargs=dict(), reuse_previous_weight_updates=False):
        old_lrs = [x['lr'] for x in list(self.optimiser.param_groups)]

        # 1. Compute \Delta W using saved weights and current weights, then divide by the learning rate to get the lr=1 update
        with torch.no_grad():
            # In-place update saved_weights to become the difference between the new weights and the saved weights
            for l, param in enumerate(self._get_parameters()):
                if not reuse_previous_weight_updates:
                    self.weight_updates[l] += param.data - 2*self.weight_updates[l] # New weights minus old weights. Extra factor of 2 because we want to remove the stored in-place value
                    param.data += -self.weight_updates[l] # reset parameter to the old values
                    self.weight_updates[l] /= old_lrs[l]

        # 2. Compute Phi, and do a backwards pass to get phigrads (dPhi/dW)
        batch = next(self.dataloader)
        _input = batch['input'].to(self.device)
        _msk = batch['mask'].to(self.device)
        modeloutput = self.model(_input, attention_mask=_msk)
        if not torch.is_tensor(modeloutput):
            modeloutput = modeloutput.logits
        self.optimiser.zero_grad()
        phi = self.phi_module(modeloutput)
        phi.backward()
        with torch.no_grad():
            phigrads = [param.grad.data for param in self._get_parameters()]

        # 3. Compute \Delta_\ell F for each layer using phigrads
        with torch.no_grad():
            delta_ell_fs = self._compute_delta_ell_fs(self.weight_updates, phigrads)

        # 4. Set new learning rates (only if modify_lrs is True)
        if not reuse_previous_weight_updates:
            if modify_lrs:
                with torch.no_grad():
                    # denom = 10. if delta_ell_fs[l].isnan() else delta_ell_fs[l].item()
                    # The 1 represents the learning rate of the original update, which we rescaled to 1 in step 0 (and is therefore unnecessary but we keep it in for clarity).
                    if self.joint_parameters is None:
                        new_lrs = [self.outerlr * self.masses[l] / (delta_ell_fs[l].item() + self.eps)  if delta_ell_fs[l] > 0. else old_lrs[l] for l in range(len(delta_ell_fs))]
                    else:
                        indiv_func_lr = {
                            name: self.masses[l] * self.outerlr
                            for l, (name, _) in enumerate(self._get_named_parameters())
                        }
                        joint_func_lrs = [
                            (indiv_func_lr[name] + sum(indiv_func_lr[jname] for jname in self.joint_parameters[name])) / (len(self.joint_parameters[name]) + 1)
                            for name, _ in self._get_named_parameters()
                        ]
                        new_lrs = [
                            joint_func_lrs[l] / delta_ell_fs[l].item()  if delta_ell_fs[l] > 0. else old_lrs[l]
                            for l in range(len(joint_func_lrs))
                        ]
                    # Update the optimiser's layerwise learning rates
                    for l, param_group in enumerate(self.optimiser.param_groups):
                        param_group["lr"] = new_lrs[l]

                # 5. Apply the normalised update (only if modify_lrs is True)
                if modify_lrs:
                    with torch.no_grad():
                        # Apply the normalised update.
                        for l, param in enumerate(self._get_parameters()):
                            param.data += self.weight_updates[l] * new_lrs[l]
                else:
                    with torch.no_grad():
                        # Reapply the original update
                        for l, param in enumerate(self._get_parameters()):
                            param.data += self.weight_updates[l] * old_lrs[l]

        self.step += 1
        # self.weight_updates = None # We don't need this anymore, free up some memory

        # Return the LR=1 delta_ell_fs if requested
        if return_delta_ell_fs:
            return delta_ell_fs

    # Estimate ||delta_\ell F||_RMS for each layer. Returns a list of scalars, one for each layer.
    def _compute_delta_ell_fs(self, updateiterator, phigraditerator):
        with torch.no_grad():
            delta_ell_fs = []
            for l in range(len(updateiterator)):
                update = updateiterator[l]
                phigrad = phigraditerator[l]

                width_bias_correction_denom = 1 - self.beta ** self.step

                # update_times_phigrad = update * phigrad
                phigrad *= update # In-place update phigrad to be the elementwise product of update and phigrad, in theory saving memory. This probably isn't important because we deal with one parameter tensor at a time.
                update_times_phigrad = phigrad # Rename for clarity

                ndim = max(update.ndim, 1) # Scalar parameters are awkward.

                if self.approx_type == "iid":
                    new_iid_var_EMA = (1-self.beta)*update_times_phigrad.pow(2).sum() + self.beta*self.iid_var_EMAs[l]
                    delta_ell_f = torch.exp(0.5*(new_iid_var_EMA.log() - math.log(width_bias_correction_denom)))
                    self.iid_var_EMAs[l] += new_iid_var_EMA - self.iid_var_EMAs[l]

                elif self.approx_type == "full_cov":
                    new_full_cov_var_EMA = (1-self.beta)*update_times_phigrad.sum().pow(2) + self.beta*self.full_cov_var_EMAs[l]
                    delta_ell_f = torch.exp(0.5*(new_full_cov_var_EMA.log() - math.log(width_bias_correction_denom)))
                    self.full_cov_var_EMAs[l] += new_full_cov_var_EMA - self.full_cov_var_EMAs[l]

                elif self.approx_type == "kronecker":
                    new_kron_var_num_EMA = torch.zeros(ndim, device=update.device, dtype=update.dtype)
                    for d in range(ndim):
                        new_kron_var_num_EMA[d] = (1-self.beta)*update_times_phigrad.sum(d).pow(2).sum() + self.beta*self.kron_var_num_EMAs[l][d]
                    new_kron_var_denom_EMA = (1-self.beta)*update_times_phigrad.pow(2).sum() + self.beta*self.kron_var_denom_EMAs[l]

                    # Compute the normaliser in log space to avoid underflow
                    # On the numerator, we have D EMAs, and on the denominator, we have D-1 EMAs, so D-1 bias corrections are cancelled out
                    delta_ell_f = torch.exp(0.5*(new_kron_var_num_EMA.log().sum() - max(0,update.ndim - 1) * new_kron_var_denom_EMA.log() -math.log(width_bias_correction_denom)))

                    self.kron_var_num_EMAs[l] += new_kron_var_num_EMA - self.kron_var_num_EMAs[l] # Use += to in-place update the EMA tensor
                    self.kron_var_denom_EMAs[l] += new_kron_var_denom_EMA - self.kron_var_denom_EMAs[l] # Use += to in-place update the EMA tensor

                if self.fix_nans:
                    if delta_ell_f.isnan():
                        assert delta_ell_f.size() == (1,)
                        delta_ell_f[0] = 0.

                delta_ell_fs.append(delta_ell_f)

            return delta_ell_fs