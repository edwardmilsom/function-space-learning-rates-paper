# This script runs a ResMLP model on the CIFAR-10 dataset, using FLeRM to either record the base FSLRs, or normalise the scaled FSLRs using previously recorded base FSLRs.
# Run first with record_basefslrs=True, then with normalise_scaled_fslrs_using_flerm=True.
# Note we have omitted code to compute loss statistics / accuracies for brevity, only saving a list of training losses.
import torch
import torch.nn as nn
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser(description='FLeRM example training script')
parser.add_argument('--width_mult', type=int, default=1, help='Width multiplier for the network')
parser.add_argument('--depth_mult', type=int, default=1, help='Depth multiplier for the network')
parser.add_argument('--block_size', type=int, default=1, help='Number of layers in each block')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--record_basefslrs', action='store_true', help='Record the base FSLRs')
parser.add_argument('--normalise_scaled_fslrs_using_flerm', action='store_true', help='Normalise the scaled FSLRs using FLeRM')
parser.add_argument('--seed', type=int, default=0, help='Seed for the random number generator')
parser.add_argument('--flerm_frequency', type=int, default=100, help='How often to record the base FSLRs')
parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
args = parser.parse_args()

if args.record_basefslrs and args.normalise_scaled_fslrs_using_flerm:
    raise ValueError("Cannot record and normalise base FSLRs in the same run")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

def seed_worker(worker_id):
    # Set fixed seed for NumPy
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    # Set fixed seed for random module
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

g2 = torch.Generator()
g2.manual_seed(args.seed+1)


# torch.set_default_dtype(torch.float64)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#%% Load the CIFAR-10 dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batchsize=256
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, worker_init_fn=seed_worker, num_workers=0, generator=g)

# Keep a separate trainloader for FLeRM, so we can give FLeRM random batches to compute its FSLR estimates
flerm_trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, worker_init_fn=seed_worker, num_workers=0, generator=g2)

#%% Define the neural network

# This is an MLP with skip connections. Depth_mult controls how many residual blocks of hidden layers there are, and width_mult controls the width of the network. Block_size controls how many layers are in each block.

class SimpleResMLP(nn.Module):
    def __init__(self, width_mult, depth_mult, block_size=1):
        super().__init__()

        self.width_mult = width_mult
        self.depth_mult = depth_mult

        base_hidden_blocks = 4
        hidden_blocks = base_hidden_blocks * depth_mult

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3072, 128*width_mult))
        
        blocks = nn.ModuleList()
        for i in range(hidden_blocks):
            blocks.append(nn.ModuleList())
            for j in range(block_size):
                blocks[i].append(nn.Linear(128*width_mult, 128*width_mult))

        self.layers.append(blocks)

        self.layers.append(nn.Linear(128*width_mult, 10))

        nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity='linear')
        nn.init.zeros_(self.layers[0].bias)

        for i in range(0, len(blocks)):
            for j in range(0, len(blocks[i])):
                nn.init.kaiming_normal_(blocks[i][j].weight, nonlinearity='relu')
                blocks[i][j].weight.data /= (depth_mult**(1/block_size))**(0.5) # Ensure each block has variance proportional to 1/(L/L0) by ensuring each layer in the block increases the variance of its input by (1/L)**(1/block_size).

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(0, len(self.layers[1])):
            x_res = x
            for j in range(0, len(self.layers[1][i])):
                x_res = torch.relu(x_res)
                x_res = self.layers[1][i][j](x_res)
                x = x + x_res
        x = self.layers[2](x)
        return x
    

#%% Prepare for training loop
    
# Define the loss function
loss_fn = nn.CrossEntropyLoss()

model = SimpleResMLP(args.width_mult, args.depth_mult, args.block_size).to(args.device)

# Define the optimizer
# Note that we put each parameter tensor into its own param group, so that FLeRM can assign layerwise learning rates.
param_groups = [{'params': [p], 'lr': args.lr} for p in model.parameters()]
optimiser = torch.optim.Adam(param_groups)

#%% Base FSLRs splitting (only relevant if depth_mult > 1, since you need to match baseFSLRs to more layers than were in the base model)

if args.normalise_scaled_fslrs_using_flerm:
    baseseed = 0
    baseFSLRs = torch.load(f"baseFSLRs_lr_{args.lr}_seed_{args.seed}.ptnorms")

# Split the base FSLRs into the correct number of layers
def generate_FSLRs_dict(step, model):
    model_par_names = [name for name, _ in model.named_parameters()]
    base_FSLRs_this_step = {name:mass_iters[step] for name, mass_iters in baseFSLRs.items()}
    base_FSLRs_names = list(base_FSLRs_this_step.keys())
    generated_FSLRs = {}
    for name in base_FSLRs_names:

        # The input and output layer are always the same as the base model
        if name in ["layers.0.weight", "layers.2.weight", "layers.0.bias", "layers.2.bias"]:
            generated_FSLRs[name] = base_FSLRs_this_step[name]

        # Since there are more hidden blocks in the new model, we need to split the FSLRs for each base block into the new blocks
        # Here we equally divide them, as in the paper, e.g. if depth_mult=4, then the base FSLR for layer n in block m is split into 4 equal parts for layer n in blocks 4m, 4m+1, 4m+2, 4m+3 (assume zero-indexing)
        elif name.startswith("layers.1."):

            # Layers in residual blocks are named like layers.1.block_number.layer_number.weight or layers.1.block_number.layer_number.bias
            old_block_num = int(name.split("layers.1.")[1].split(".")[0])
            new_block_num = args.depth_mult*old_block_num
            for d in range(args.depth_mult):
                # Note that by applying this logic to the block number (and not touching the layer number), we automatically handle the structure within each block
                param_name = name.replace(f"layers.1.{old_block_num}.", f"layers.1.{new_block_num+d}.")
                generated_FSLRs[param_name] = base_FSLRs_this_step[name] / args.depth_mult
        else:
            raise ValueError(f"Name {name} not recognised")
    
    for name in model_par_names:
        assert name in generated_FSLRs, f"Parameter {name} from model not in generated FSLRs"
    for name in generated_FSLRs:
        assert name in model_par_names, f"Parameter {name} from generated FSLRs not in model par names"
    
    return generated_FSLRs

split_FSLRs = {}
if args.normalise_scaled_fslrs_using_flerm:
    split_FSLRs = generate_FSLRs_dict(0, model)


#%% FLeRM setup

def model_output_closure(X):
    return model(X)

from flerm import FLeRM
flerm = FLeRM(model_output_closure, optimiser, args.lr, model.named_parameters(), baseFSLRs = split_FSLRs)

# Keep a list of recorded FSLRs for each parameter tensor.
if args.record_basefslrs:
    recordedFSLRs_iters_dict = {name:[] for name, param in model.named_parameters()}


#%% Training loop

train_losses = []

iteration = 0

# Logic to reset the flerm dataloader in case it runs out of data
flerm_trainloader_iter = iter(flerm_trainloader)
def get_next_flerm_batch():
    global flerm_trainloader_iter
    try:
        flerm_X, _ = next(flerm_trainloader_iter)
    except StopIteration:
        flerm_trainloader_iter = iter(flerm_trainloader)
        flerm_X, _ = next(flerm_trainloader_iter)
    return flerm_X

for i in range(5):
    for images, labels in trainloader:
        
        X, y = images.flatten(1,3).to(args.device), labels.to(args.device)

        optimiser.zero_grad()

        # FLeRM requires us to save the weights before the backward pass happens, so it can calculate the update that was performed as new_weights - old_weights
        # We only use FLeRM every args.flerm_frequency iterations
        if (args.record_basefslrs or args.normalise_scaled_fslrs_using_flerm) and iteration % args.flerm_frequency == 0:
            flerm.save_weights()
            if args.normalise_scaled_fslrs_using_flerm:
                split_FSLRs = generate_FSLRs_dict(iteration//args.flerm_frequency, model)
                flerm.set_baseFSLRs(split_FSLRs)
        
        y_pred = model(X)
        loss = loss_fn(y_pred.squeeze(), y.long())
        loss.backward()
        optimiser.step()

        # OPTIONAL: Warmup the FLeRM EMA estimates by running its estimates on lots of batches. This is not strictly necessary, but can help stabilise the estimates.
        # See below "Regular FLERM step" for the actual FLeRM step and explanation of the update_lrs function.
        if iteration == 0 and (args.record_basefslrs or args.normalise_scaled_fslrs_using_flerm):
            # Run FLeRM first time so it can compute weight updates as new_weights - old_weights
            flerm_X = get_next_flerm_batch().flatten(1,3).to(args.device)
            flerm.update_lrs(flerm_X, modify_lrs=False)
            # Run FLeRM on a few more batches to warm up the EMA estimates
            # This time, don't let FLeRM try to recompute the weight updates, since it won't work anymore because we use inplace operations to save memory
            for _ in range(39):
                flerm_X = get_next_flerm_batch().flatten(1,3).to(args.device)
                flerm.update_lrs(flerm_X, modify_lrs=False, reuse_previous_weight_updates=True)
            # Regular FLeRM step (same as below, see below for explanation) except we need to use the reuse_previous_weight_updates flag to avoid recomputing the weight updates
                flerm_X = get_next_flerm_batch().flatten(1,3).to(args.device)
                if args.record_basefslrs:
                    baseFSLRsthisstep = flerm.update_lrs(flerm_X, modify_lrs=False, return_delta_ell_fs=True, reuse_previous_weight_updates=True)
                    parindex=0
                    for paramname, _ in model.named_parameters():
                        recordedFSLRs_iters_dict[paramname].append(baseFSLRsthisstep[parindex])
                        parindex+=1
                elif args.normalise_scaled_fslrs_using_flerm:
                    flerm.update_lrs(flerm_X, modify_lrs=True, return_delta_ell_fs=False, reuse_previous_weight_updates=True)

        # When not on the first iteration / warming up FLeRM, we can run the regular FLeRM step
        # Regular FLeRM step (where we actually update the learning rates, or record the base FSLRs)
        elif iteration % args.flerm_frequency == 0 and (args.record_basefslrs or args.normalise_scaled_fslrs_using_flerm):
            flerm_X = get_next_flerm_batch().flatten(1,3).to(args.device)
            # RECORDING BASE FSLRS
            if args.record_basefslrs:
                # Use the update_lrs function to record the FSLRs for each parameter tensor (modify_lrs=False means we don't change the optimiser learning rates, and return_delta_ell_fs=True means we return the FSLRs we estimated)
                # Note that the update_lrs function temporarily modifies the model weights (subtracts the computed update), runs a forward pass, then a backwards pass (on the random linear combination of the outputs), computes its FSLR estimates, then restores the model weights.
                # Note that the base FSLRs are returned as a list of floats, one for each parameter tensor, in the same order as model.named_parameters() (there is no good reason for not returning a dict, it's just how it was implemented).
                baseFSLRsthisstep = flerm.update_lrs(flerm_X, modify_lrs=False, return_delta_ell_fs=True)
                parindex=0
                for paramname, _ in model.named_parameters():
                    recordedFSLRs_iters_dict[paramname].append(baseFSLRsthisstep[parindex])
                    parindex+=1
            # NORMALISING SCALED FSLRS USING FLeRM
            elif args.normalise_scaled_fslrs_using_flerm:
                # Use the update_lrs function to update the learning rates of the optimiser, using the base FSLRs we provided (modify_lrs=True means we do change the optimiser paramgroup learning rates, and return_delta_ell_fs=False means we don't return the FSLRs we estimated for this scaled model)
                # Note that the update_lrs function modifies the model weights (subtracts the computed update), runs a forward pass, then a backwards pass (on the random linear combination of the outputs), computes its FSLR estimates, then restores the model weights AS IF THEY WERE USING THE NEW NORMALISED LEARNING RATES IN THAT STEP.

                flerm.update_lrs(flerm_X, modify_lrs=True, return_delta_ell_fs=False)

        train_losses.append(loss.item())
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, loss: {loss.item() / batchsize}") # Divide by batch size to get per-sample loss
        
        # Not implemented to keep code short: compute loss statistics / accuracies here
        #compute_metrics()
        
        iteration += 1

#%% Save the base FSLRs
if args.record_basefslrs:
    torch.save(recordedFSLRs_iters_dict, f"baseFSLRs_lr_{args.lr}_seed_{args.seed}.ptnorms")