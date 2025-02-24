
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

# Function-space Learning Rate Matching
class FLeRM(nn.Module):
    def _get_named_parameters(self):
        return iter((name, param) for name, param in self._named_parameters)
    
    def _get_parameters(self):
        return iter(param for name, param in self._named_parameters)
    
    # Should we make the user change beta based on how often they call this, or should we make them provide the number of iterations between calls?
    def __init__(self, modeloutputclosure, optimiser, outerlr, named_parameters, baseFSLRs=None, eps=0, beta=0.9, approx_type="kronecker",):
        super().__init__()

        self.modeloutputclosure = modeloutputclosure
        self.optimiser = optimiser
        
        self._named_parameters = list(named_parameters)

        self.set_baseFSLRs(baseFSLRs)

        self.device = next(self._get_parameters()).device
        self.dtype = next(self._get_parameters()).dtype

        #Initialise stuff
        self.step = 1
        self.eps = eps
        self.beta = beta
        self.approx_type = approx_type
        self.phi_module = RandomLinComb()
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
        self.aggregate_var_EMA = torch.zeros(1, device=self.device, dtype=self.dtype)

    # baseFSLRs is a dictionary of the form {parameter_name: baseFSLR}, assumed to contain all parameters in the model.
    # Controls contribution of each parameter to total function update, i.e. \Delta F AFTER normalisation.
    def set_baseFSLRs(self, baseFSLRs):

        num_params = sum(1 for _ in self._get_parameters())

        # If no baseFSLRs are given, initialise them all equally. Suboptimal for almost all cases.
        if baseFSLRs is None or baseFSLRs == {}:
            named_parameters = dict(self._get_named_parameters())
            self.baseFSLRs = [1/num_params for _ in range(num_params)]
            print("Warning: No baseFSLRs provided (ignore this warning if recording base FSLRs). If normalising scaled FSLRs, all parameters will be updated with equal baseFSLR adding to 1, which is suboptimal for almost all cases. To surpress this warning, go into the code and comment it out ;)")

        else:
            # If baseFSLRs is not a dict, throw an error
            if not isinstance(baseFSLRs, dict):
                raise ValueError("Provided baseFSLRs must be a dictionary specifying the baseFSLR for all the named parameters. Use the same names as in model.named_parameters().")

            # Check that all the keys in baseFSLRs are in the model
            named_parameters = dict(self._get_named_parameters())
            for name in baseFSLRs:
                if name not in named_parameters:
                    raise ValueError(f"Parameter {name} not found in model")

            # Construct the baseFSLRs list, spreading the remaining baseFSLR equally among the parameters not specified in baseFSLRs
            self.baseFSLRs = [baseFSLRs[name] for name in named_parameters]
            assert len(self.baseFSLRs) == num_params

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
    def update_lrs(self, batchinputs, return_delta_ell_fs=False, modify_lrs=True, reuse_previous_weight_updates=False, preserve_exact_copy_of_weights=False):

        # 1. Compute \Delta W using saved weights and current weights, then divide by the learning rate to get the lr=1 update
        with torch.no_grad():
            
            # Save the exact copy of the weights if requested (otherwise small numerical errors can occur). Note this uses more memory.
            if preserve_exact_copy_of_weights:
                saved_weights = [param.data.clone() for param in self._get_parameters()]

            # In-place update saved_weights to become the difference between the new weights and the saved weights
            for l, param in enumerate(self._get_parameters()):
                if not reuse_previous_weight_updates:
                    self.weight_updates[l] += param.data - 2*self.weight_updates[l] # New weights minus old weights. Extra factor of 2 because we want to remove the stored in-place value

                # Undo the last update on the weights. If we are reusing previous weight updates, we need to put the learning rate back in for this step.
                if not reuse_previous_weight_updates:
                    param.data += -self.weight_updates[l]
                else:
                    param.data += -self.weight_updates[l] * list(self.optimiser.param_groups)[l]["lr"]

                # Divide by the learning rate to get the lr=1 update
                # This is very important, as we use exponential moving averages to estimate the variance of update*phigrad, so update's magnitude should be kept constant
                # If you don't ensure a fixed learning rate, the exponential moving average will be trying to estimate a moving target
                # Note that this assumes an optimiser like Adam or SignSGD where updates have size ||\Delta W||_RMS = 1
                if not reuse_previous_weight_updates:
                    self.weight_updates[l] /= list(self.optimiser.param_groups)[l]["lr"]
            
        # 2. Compute Phi, and do a backwards pass to get phigrads (dPhi/dW)
        modeloutput = self.modeloutputclosure(batchinputs)
        
        self.optimiser.zero_grad()
        phi = self.phi_module(modeloutput)
        phi.backward()
        with torch.no_grad():
            phigrads = [param.grad.data for param in self._get_parameters()]

        # 3. Compute \Delta_\ell F for each layer using phigrads
        with torch.no_grad():
            delta_ell_fs = self._compute_delta_ell_fs(self.weight_updates, phigrads)
        
        # 4. Set new learning rates (only if modify_lrs is True)
        if modify_lrs:
            with torch.no_grad():
                # The 1 represents the learning rate of the original update, which we rescaled to 1 in step 0 (and is therefore unnecessary but we keep it in for clarity).
                new_lrs = [1 * self.outerlr * (self.baseFSLRs[l] + self.eps) / (delta_ell_fs[l] + self.eps) for l in range(len(delta_ell_fs))]
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
                    param.data += self.weight_updates[l] * list(self.optimiser.param_groups)[l]["lr"]

        self.step += 1
        # self.weight_updates = None # We don't need this anymore, free up some memory. I'm not sure if this actually affects peak memory.

        # Restore the exact copy of the weights if requested
        if preserve_exact_copy_of_weights:
            for l, param in enumerate(self._get_parameters()):
                param.data.copy_(saved_weights[l])
        
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

                delta_ell_fs.append(delta_ell_f.item())

            return delta_ell_fs