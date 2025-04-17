import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR

# Create a device argument
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--datafolder', type=str, default='../data/')
parser.add_argument('--width_mult', type=int, default=1,
                    help='width multiplier for the transformer model')
parser.add_argument('--depth_mult', type=int, default=1,
                    help='depth multiplier for the transformer model')
parser.add_argument('--optimiser', type=str, default='adam', choices=['adam', 'sgd', 'sgdmomentum'])
parser.add_argument('--normalised', action='store_true', help='Use normalised update')
parser.add_argument('--normaliser_update_frequency', type=int, default=100)
parser.add_argument('--save_name', type=str, help='Path of the file to save the results to, usually with a .pt extension. E.g. \"yourfilename.pt\"')
parser.add_argument('--ptprefix', type=str, default='', help='Prefix for the save .pt file at the end of the run')
parser.add_argument('--ptnormsprefix', type=str, default='', help='Prefix for the save .ptnorm file for the measured masses')
parser.add_argument('--normaliser_beta', type=float, default=0.9)
parser.add_argument('--normaliser_approx_type', type=str, default='kronecker', choices=['kronecker', 'iid', 'full_cov'])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--only_measure_masses', action='store_true', help='Only measure the masses using the normaliser, don\'t actually change the learning rates')
parser.add_argument('--block_size', type=int, default=1, help='Number of layers in each block')
parser.add_argument('--model', type=str, default='resmlp', choices=['mlp', 'resmlp'])
parser.add_argument('--use_forward_pass_rootL', action='store_true', help='Use the forward pass root L normalisation')
parser.add_argument('--equal_mass_ablation', action='store_true', help='Use equal masses for all parameters')
parser.add_argument('--only_flerm_first_step_ablation', action='store_true', help='Only use the flerm for the first step of the normaliser update')
parser.add_argument('--equal_mass_but_still_splitting_depth_properly_ablation', action='store_true', help='Use equal masses for all parameters, but still split the masses properly for the depth multiplier')
parser.add_argument('--use_scheduler', action='store_true', help='Use a scheduler for the learning rate')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
import random
import numpy as np
import os
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# torch.set_default_dtype(torch.float64)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Weights and Biases init
# import wandb
# if args.only_measure_masses:
#     wandb.init(
#             project="measuremasses_mlp_cifar10",
#             config={
#                 "learning_rate": args.lr,
#                 "depth_multiplier": args.depth_mult,
#                 "width_multiplier": args.width_mult,
#                 "seed": args.seed,
#                 "normalised": args.normalised,
#                 "optimiser": args.optimiser,
#                 "architecture": "MLP",
#                 "dataset": "CIFAR10",
#                 "epochs": 1,
#             },
#             # Organize runs with groups and job types
#             group=f"{args.ptnormsprefix}_{args.ptprefix}_normalised_{args.normalised}_optimiser_{args.optimiser}_lr_{args.lr}",  # group by depthmult
#             name=f"{args.ptnormsprefix}_{args.ptprefix}_normalised_{args.normalised}_optimiser_{args.optimiser}_lr_{args.lr}_depthmult_{args.depth_mult}_widthmult_{args.width_mult}_seed_{args.seed}"  # custom run name
#         )
# else:
#     wandb.init(
#                 project="mlp_cifar10",
#                 config={
#                     "learning_rate": args.lr,
#                     "depth_multiplier": args.depth_mult,
#                     "width_multiplier": args.width_mult,
#                     "seed": args.seed,
#                     "normalised": args.normalised,
#                     "optimiser": args.optimiser,
#                     "architecture": "MLP",
#                     "dataset": "CIFAR10",
#                     "epochs": 1,
#                 },
#                 # Organize runs with groups and job types
#                 group=f"{args.ptnormsprefix}_{args.ptprefix}_normalised_{args.normalised}_optimiser_{args.optimiser}_lr_{args.lr}",  # group by depthmult
#                 name=f"{args.ptnormsprefix}_{args.ptprefix}_normalised_{args.normalised}_optimiser_{args.optimiser}_lr_{args.lr}_depthmult_{args.depth_mult}_widthmult_{args.width_mult}_seed_{args.seed}"  # custom run name
#             )

#%% Load the CIFAR-10 dataset

# Load the CIFAR-10 dataset
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Download and load the CIFAR-10 training dataset
trainset = datasets.CIFAR10(root=args.datafolder, train=True, download=True, transform=transform)

# Download and load the CIFAR-10 test dataset
testset = datasets.CIFAR10(root=args.datafolder, train=False, download=True, transform=transform)

#%% Custom dataloader

class InMemoryCIFAR10(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images.to(args.device)
        self.labels = labels.to(args.device)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

full_trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainset.data.shape[0], shuffle=True)
train_images, train_labels = next(iter(full_trainloader))

in_memory_train_set = InMemoryCIFAR10(train_images, train_labels)
in_memory_train_loader = torch.utils.data.DataLoader(in_memory_train_set, batch_size=256, shuffle=True)
flerm_in_memory_train_loader = torch.utils.data.DataLoader(in_memory_train_set, batch_size=256, shuffle=True)

# Set in_memory_train_loader to a normal data loader
# in_memory_train_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)


test_loader = torch.utils.data.DataLoader(testset, batch_size=testset.data.shape[0], shuffle=True)
test_images, test_labels = next(iter(test_loader))

in_memory_test_set = InMemoryCIFAR10(test_images, test_labels)

#%% Model

# Define a simple 1 hidden layer neural network
class SimpleMLP(nn.Module):
    def __init__(self, width_mult, depth_mult):
        super().__init__()

        self.width_mult = width_mult
        self.depth_mult = depth_mult

        # Base model
        # self.fc1 = nn.Linear(3072, 256*width_mult, bias=True)
        # self.fc2 = nn.Linear(256*width_mult, 256*width_mult, bias=True) # depth_mult of these
        # self.fc3 = nn.Linear(256*width_mult, 10, bias=True)

        # nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='linear')
        # nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

        # Generate the model using the above blueprint
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(3072, 128*width_mult, bias=True))
        for i in range(depth_mult):
            self.linear_layers.append(nn.Linear(128*width_mult, 128*width_mult, bias=True))
        self.linear_layers.append(nn.Linear(128*width_mult, 10, bias=True))

        nn.init.kaiming_normal_(self.linear_layers[0].weight, nonlinearity='linear')
        nn.init.zeros_(self.linear_layers[0].bias)
        for i in range(1, len(self.linear_layers)):
            nn.init.kaiming_normal_(self.linear_layers[i].weight, nonlinearity='relu')
            nn.init.zeros_(self.linear_layers[i].bias)

    def forward(self, x):

        # x = self.fc1(x)
        # x = torch.relu(x)
        # x = self.fc2(x)
        # x = torch.relu(x)
        # x = self.fc3(x)
        # return x

        for i in range(len(self.linear_layers)-1):
            x = self.linear_layers[i](x)
            x = torch.relu(x)
        x = self.linear_layers[-1](x)
        return x

class SimpleResMLP(nn.Module):
    def __init__(self, width_mult, depth_mult, block_size=1, use_forward_pass_rootL=False):
        super().__init__()

        self.width_mult = width_mult
        self.depth_mult = depth_mult

        self.use_forward_pass_rootL = use_forward_pass_rootL

        base_hidden_layers = 4
        hidden_layers = base_hidden_layers * depth_mult

        # Base model
        # self.fc1 = nn.Linear(3072, 128*width_mult, bias=True)
        # self.fc2 = nn.Linear(128*width_mult, 128*width_mult, bias=True) # depth_mult of these
        # self.fc3 = nn.Linear(128*width_mult, 10, bias=True)

        # nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='linear')
        # nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')

        # Generate the model using the above blueprint
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3072, 128*width_mult))
        
        blocks = nn.ModuleList()
        for i in range(hidden_layers):
            blocks.append(nn.ModuleList())
            for j in range(block_size):
                blocks[i].append(nn.Linear(128*width_mult, 128*width_mult))

        self.layers.append(blocks)

        self.layers.append(nn.Linear(128*width_mult, 10))

        nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity='linear')
        nn.init.zeros_(self.layers[0].bias)

        for i in range(0, len(blocks)):
            # nn.init.kaiming_normal_(self.linear_layers[i].weight, nonlinearity='relu')
            # self.linear_layers[i].weight.data /= hidden_layers**0.5
            # nn.init.zeros_(self.linear_layers[i].bias)
            for j in range(0, len(blocks[i])):
                nn.init.kaiming_normal_(blocks[i][j].weight, nonlinearity='relu')
                if not self.use_forward_pass_rootL:
                    blocks[i][j].weight.data /= (depth_mult**(1/block_size))**(0.5) # Ensure each block has variance proportional to 1/(L/L0) by ensuring each layer in the block increases the variance of its input by (1/L)**(1/block_size).

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(0, len(self.layers[1])):
            x_res = x
            for j in range(0, len(self.layers[1][i])):
                x_res = torch.relu(x_res)
                x_res = self.layers[1][i][j](x_res)
            if self.use_forward_pass_rootL:
                x = x + (1/self.depth_mult**0.5) * x_res
            else:
                x = x + x_res
        x = self.layers[2](x)
        return x

# class SimpleResMLP(nn.Module):
#     def __init__(self, width_mult, depth_mult, block_size=1):
#         super().__init__()

#         self.width_mult = width_mult
#         self.depth_mult = depth_mult

#         base_hidden_layers = 1
#         hidden_layers = base_hidden_layers * depth_mult

#         # Generate the model using the above blueprint
#         self.layers = nn.ModuleList()
#         self.layers.append(nn.Linear(3072, 128*width_mult))
        
#         blocks = nn.ModuleList()
#         for i in range(hidden_layers):
#             blocks.append(nn.ModuleList())
#             for j in range(block_size):
#                 blocks[i].append(nn.Linear(128*width_mult, 128*width_mult))

#         self.layers.append(blocks)

#         self.layers.append(nn.Linear(128*width_mult, 10))

#         # nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity='linear')
#         nn.init.orthogonal_(self.layers[0].weight)
#         nn.init.zeros_(self.layers[0].bias)

#         # nn.init.kaiming_normal_(self.layers[2].weight, nonlinearity='linear')
#         nn.init.orthogonal_(self.layers[2].weight)
#         nn.init.zeros_(self.layers[2].bias)

#         for i in range(0, len(blocks)):
#             for j in range(0, len(blocks[i])):
#                 # nn.init.kaiming_normal_(blocks[i][j].weight, nonlinearity='relu')
#                 nn.init.orthogonal_(blocks[i][j].weight)
#                 nn.init.zeros_(blocks[i][j].bias)
        
#         self.layernormaliser = nn.LayerNorm(128*width_mult, elementwise_affine=False)

#     def forward(self, x):
#         x = self.layers[0](x)
#         for i in range(0, len(self.layers[1])):
#             x_res = self.layernormaliser(x)
#             for j in range(0, len(self.layers[1][i])):
#                 x_res = torch.relu(x_res)
#                 x_res = self.layers[1][i][j](x_res)
#             x = x + x_res
#         x = self.layers[2](x)
#         return x

# widths = [128,256,512,1024,2048,4096]
# widths = [2048]
# Create 10 equally spaced log-learning rates between 10 and 0.01
# log_lrs = torch.linspace(math.log(1), math.log(0.1), 10) # Normaliser version
# log_lrs = torch.linspace(math.log(0.1), math.log(0.1), 1)
# log_lrs = torch.linspace(math.log(0.01), math.log(0.00001), 10) # Adam (attempt 2)
# log_lrs = torch.linspace(math.log(1), math.log(0.0001), 10) # SGD
# lrs_torch = torch.exp(log_lrs)
# lrs = lrs_torch.tolist()

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

epochs = 50

#flerm is in the parent directory, so we need to import it from ..
os.sys.path.append("..")
from flerm import FLeRM

if args.model == 'mlp':
    model = SimpleMLP(args.width_mult, args.depth_mult).to(args.device)
elif args.model == 'resmlp':
    model = SimpleResMLP(args.width_mult, args.depth_mult, args.block_size, args.use_forward_pass_rootL).to(args.device)

# if args.normalised:
#     param_groups = [{'params': [p], 'lr': 1} for p in model.parameters()]
# else:
#     param_groups = [{'params': [p], 'lr': args.lr} for p in model.parameters()]

param_groups = [{'params': [p], 'lr': args.lr} for p in model.parameters()] # We don't really care about the learning rate here as long as it's sensible. The normaliser will reset it on the first iteration.

if args.optimiser == 'adam':
    optimiser = torch.optim.Adam(param_groups)
elif args.optimiser == 'sgd':
    optimiser = torch.optim.SGD(param_groups)
elif args.optimiser == 'sgdmomentum':
    optimiser = torch.optim.SGD(param_groups, momentum=0.9)

dummyparameterforscheduler = torch.nn.Parameter(torch.tensor(1.0))
dummyoptimiserforscheduler = torch.optim.Adam([dummyparameterforscheduler], lr=args.lr)

if args.equal_mass_ablation and args.equal_mass_but_still_splitting_depth_properly_ablation:
    raise ValueError("Can't have both equal_mass_ablation and equal_mass_but_still_splitting_depth_properly_ablation as True")

if args.only_measure_masses or not args.normalised or args.equal_mass_ablation:
    masses = {}
elif args.normalised:
    # Fetch observed masses from the training runs, and average them over the seeds
    seeds = [0,1,2,3,4,5,6,7]
    # seeds = [args.seed]
    # seeds = [0]
    obs_masses_avg_seeds_dict_inited = False
    obs_masses_avg_seeds_dict = {}
    if not args.equal_mass_but_still_splitting_depth_properly_ablation:
        for seed in seeds:
            single_seed_observed_masses_training_dict = torch.load(f"{args.ptnormsprefix}basemodel{args.model}cifar10empiricalmasses_lr_{args.lr}_seed_{seed}.ptnorms")
            for key in single_seed_observed_masses_training_dict:
                for i in range(len(single_seed_observed_masses_training_dict[key])):
                    single_seed_observed_masses_training_dict[key][i] = single_seed_observed_masses_training_dict[key][i]
            if not obs_masses_avg_seeds_dict_inited:
                obs_masses_avg_seeds_dict = single_seed_observed_masses_training_dict
                obs_masses_avg_seeds_dict_inited = True
            else:
                for key in single_seed_observed_masses_training_dict:
                    for i in range(len(single_seed_observed_masses_training_dict[key])):
                        obs_masses_avg_seeds_dict[key][i] += single_seed_observed_masses_training_dict[key][i]
        for key in obs_masses_avg_seeds_dict:
            for i in range(len(obs_masses_avg_seeds_dict[key])):
                obs_masses_avg_seeds_dict[key][i] /= len(seeds)
    else:
        example_mass_file = torch.load("weightInitRootLbasemodelresmlpcifar10empiricalmasses_lr_0.00013593626172015485_seed_0.ptnorms")
        base_param_names = list(example_mass_file.keys())
        obs_masses_avg_seeds_dict = {name:[] for name in base_param_names}
    
    if args.model == "resmlp":
        def generate_masses_dict(step, model):
            model_par_names = [name for name, _ in model.named_parameters()]
            if args.equal_mass_but_still_splitting_depth_properly_ablation:
                observed_masses_this_step = {name:1/len(obs_masses_avg_seeds_dict.keys()) for name, mass_iters in obs_masses_avg_seeds_dict.items()}
            else:
                observed_masses_this_step = {name:mass_iters[step] for name, mass_iters in obs_masses_avg_seeds_dict.items()}
            observed_masses_names = list(observed_masses_this_step.keys())
            generated_masses = {}
            for name in observed_masses_names:
                if name in ["layers.0.weight", "layers.2.weight", "layers.0.bias", "layers.2.bias"]:
                    generated_masses[name] = observed_masses_this_step[name]
                elif name.startswith("layers.1."):
                    # Layers in residual blocks are named like layers.1.block_number.layer_number.weight or layers.1.block_number.layer_number.bias
                    old_block_num = int(name.split("layers.1.")[1].split(".")[0])
                    new_block_num = args.depth_mult*old_block_num
                    for d in range(args.depth_mult):
                        param_name = name.replace(f"layers.1.{old_block_num}.", f"layers.1.{new_block_num+d}.")
                        generated_masses[param_name] = observed_masses_this_step[name] / args.depth_mult # Note that by applying this logic to the block number, we automatically copy all masses for layers within blocks to the other layers
                else:
                    raise NotImplementedError(f"Name {name} not recognised")
            
            for name in model_par_names:
                assert name in generated_masses, f"Name {name} not in generated masses"
            for name in generated_masses:
                assert name in model_par_names, f"Name {name} not in model par names"
            
            return generated_masses
    # Note this masses code assumes the base model only has 1 hidden layer. Will not work if you change the base model.
    elif args.model == "mlp":
        def generate_masses_dict(step, model):
            model_par_names = [name for name, _ in model.named_parameters()]
            if args.equal_mass_but_still_splitting_depth_properly_ablation:
                observed_masses_this_step = {name:1/len(obs_masses_avg_seeds_dict.keys()) for name, mass_iters in obs_masses_avg_seeds_dict.items()}
            else:
                observed_masses_this_step = {name:mass_iters[step] for name, mass_iters in obs_masses_avg_seeds_dict.items()}
            observed_masses_names = list(observed_masses_this_step.keys())
            generated_masses = {}
            for name in observed_masses_names:
                if name in ["linear_layers.0.weight", "linear_layers.0.bias"]:
                    generated_masses[name] = observed_masses_this_step[name]
                elif name in ["linear_layers.2.weight", "linear_layers.2.bias"]:
                    param_name = name.replace("linear_layers.2.", f"linear_layers.{args.depth_mult+1}.")
                    generated_masses[param_name] = observed_masses_this_step[name]
                elif name.startswith("linear_layers.1"):
                    for d in range(args.depth_mult):
                        param_name = name.replace(f"linear_layers.1.", f"linear_layers.{d+1}.") # Add 1 back on because we subtracted 1 earlier
                        generated_masses[param_name] = observed_masses_this_step[name] / args.depth_mult # Note that by applying this logic to the block number, we automatically copy all masses for layers within blocks to the other layers
                else:
                    raise NotImplementedError(f"Name {name} not recognised")
            
            for name in model_par_names:
                assert name in generated_masses, f"Name {name} not in generated masses"
            for name in generated_masses:
                assert name in model_par_names, f"Name {name} not in model par names"
            
            return generated_masses

    masses = generate_masses_dict(0, model)
else:
    raise NotImplementedError("Didn't expect to get here, please debug")

# Apply an EMA to the observed masses, using args.normaliser_beta as the decay rate
# This is to smooth out the observed masses, as they can be quite noisy
# for key in obs_masses_avg_seeds_dict:
#     for i in range(len(obs_masses_avg_seeds_dict[key])):
#         if i == 0:
#             obs_masses_avg_seeds_dict[key][i] = obs_masses_avg_seeds_dict[key][i] # Init the EMA at the first value
#         else:
#             obs_masses_avg_seeds_dict[key][i] = obs_masses_avg_seeds_dict[key][i-1] * args.normaliser_beta + obs_masses_avg_seeds_dict[key][i] * (1 - args.normaliser_beta) 


def model_output_closure(X):
    return model(X)

# Initialise the normaliser
normaliser = FLeRM(model_output_closure, optimiser, args.lr, model.named_parameters(), beta=args.normaliser_beta, approx_type=args.normaliser_approx_type, baseFSLRs = masses)

import time

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

iteration = 0
intermediate_running_total_loss = 0
intermediate_running_total_correct = 0
intermediate_running_total_datapoints = 0
outer_total_loss = 0
outer_total_correct = 0
outer_total_datapoints = 0

if args.only_measure_masses:
    normalisers_iters_dict = {name:[] for name, param in model.named_parameters()}

flerm_in_memory_train_loader_iter = iter(flerm_in_memory_train_loader)

# cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
for i in range(epochs):
    start_time = time.time()
    for images, labels in in_memory_train_loader: #There seems to be some kind of memory leak here, possibly when using in_memory_train_loader, but also to a lesser extent when just using trainloader?
        
        # X, y = images.flatten(1,3).to(args.device), labels.to(args.device).float()
        X, y = images.flatten(1,3), labels

        optimiser.zero_grad()

        if args.normalised and iteration % args.normaliser_update_frequency == 0:
            normaliser.save_weights() # Save the weights before the update
            if not args.only_measure_masses and not args.equal_mass_ablation:
                normaliser.set_baseFSLRs(generate_masses_dict(iteration//args.normaliser_update_frequency, model))

        y_pred = model(X)
        
        loss = loss_fn(y_pred.squeeze(), y.long())
        loss.backward()
        optimiser.step()
        
        # Do 40 warmup iterations at the start of training (run the normaliser for a few batches without updating the learning rates)
        # Note: normaliser.update_lrs usually in_place overwrites the tensor created in normaliser.save_weights() to become the updates to the weights, so we need a flag reuse_previous_weight_updates=True to prevent this when doing warmup, as the optimiser step isn't changing.
        if iteration == 0 and args.use_scheduler:
            dummyscheduler = CosineAnnealingLR(dummyoptimiserforscheduler, T_max=10000,eta_min=1e-6)
        
        if iteration == 0 and args.normalised:
            # Run first time to compute weight_updates
            try:
                flerm_X, _ = next(flerm_in_memory_train_loader_iter)
            except StopIteration:
                flerm_in_memory_train_loader_iter = iter(flerm_in_memory_train_loader)
                flerm_X, _ = next(flerm_in_memory_train_loader_iter)
            flerm_X = flerm_X.flatten(1,3)
            normaliser.update_lrs(flerm_X, modify_lrs=False)
            for _ in range(39):
                try:
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                except StopIteration:
                    flerm_in_memory_train_loader_iter = iter(flerm_in_memory_train_loader)
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                flerm_X = flerm_X.flatten(1,3)
                normaliser.update_lrs(flerm_X, modify_lrs=False, reuse_previous_weight_updates=True)
            
            if args.normalised and iteration % args.normaliser_update_frequency == 0 and not args.only_measure_masses:
                try:
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                except StopIteration:
                    flerm_in_memory_train_loader_iter = iter(flerm_in_memory_train_loader)
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                flerm_X = flerm_X.flatten(1,3)
                normaliser.update_lrs(flerm_X, reuse_previous_weight_updates=True) # Replace the last update with the normalised update, and update the learning rates for subsequent updates.
            elif args.normalised and iteration % args.normaliser_update_frequency == 0 and args.only_measure_masses:
                try:
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                except StopIteration:
                    flerm_in_memory_train_loader_iter = iter(flerm_in_memory_train_loader)
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                flerm_X = flerm_X.flatten(1,3)
                # if (flerm_X == X).all():
                #     print("WARNING: flerm_X is the same as X. This should be extremely unlikely.")
                normalisers = normaliser.update_lrs(flerm_X, modify_lrs=False, return_delta_ell_fs=True, reuse_previous_weight_updates=True) # Instead of actually changing the LRs, just return what the estimate dF caused by each parameter's update is (for LR=1).
                namedparam_normaliser_dict = {namedparam: nmlsr for namedparam, nmlsr in zip(model.named_parameters(), normalisers)}
                # Print name: normaliser for each parameter
                for namedparam, nmlsr in namedparam_normaliser_dict.items():
                #     print(f"{namedparam[0]}: {nmlsr}")
                    normalisers_iters_dict[namedparam[0]].append(nmlsr)

        # On all other iterations, just do one FLeRM step
        elif not args.only_flerm_first_step_ablation:
            if args.normalised and iteration % args.normaliser_update_frequency == 0 and not args.only_measure_masses:
                try:
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                except StopIteration:
                    flerm_in_memory_train_loader_iter = iter(flerm_in_memory_train_loader)
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                flerm_X = flerm_X.flatten(1,3)
                normaliser.update_lrs(flerm_X) # Replace the last update with the normalised update, and update the learning rates for subsequent updates.
            elif args.normalised and iteration % args.normaliser_update_frequency == 0 and args.only_measure_masses:
                try:
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                except StopIteration:
                    flerm_in_memory_train_loader_iter = iter(flerm_in_memory_train_loader)
                    flerm_X, _ = next(flerm_in_memory_train_loader_iter)
                flerm_X = flerm_X.flatten(1,3)
                # if (flerm_X == X).all():
                #     print("WARNING: flerm_X is the same as X. This should be extremely unlikely.")
                normalisers = normaliser.update_lrs(flerm_X, modify_lrs=False, return_delta_ell_fs=True) # Instead of actually changing the LRs, just return what the estimate dF caused by each parameter's update is (for LR=1).
                namedparam_normaliser_dict = {namedparam: nmlsr for namedparam, nmlsr in zip(model.named_parameters(), normalisers)}
                # Print name: normaliser for each parameter
                for namedparam, nmlsr in namedparam_normaliser_dict.items():
                #     print(f"{namedparam[0]}: {nmlsr}")
                    normalisers_iters_dict[namedparam[0]].append(nmlsr)

        batch_correct = (torch.argmax(y_pred, dim=1) == y).sum().item()

        intermediate_running_total_loss += loss.detach().clone().item() * y.size(0)
        intermediate_running_total_correct += batch_correct
        intermediate_running_total_datapoints += y.size(0)

        logging_interval = 200
        if iteration % logging_interval == 0:
            avg_train_loss = intermediate_running_total_loss / (logging_interval * intermediate_running_total_datapoints)
            avg_train_accuracy = intermediate_running_total_correct / intermediate_running_total_datapoints
            
            model.eval()

            y_pred_test = model(in_memory_test_set.images.flatten(1,3))
            _, predicted_test = torch.max(y_pred_test, 1)
            correct_test = (predicted_test == in_memory_test_set.labels).sum().item()
            test_accuracy = correct_test / test_labels.size(0)
            test_loss = loss_fn(y_pred_test.squeeze(), test_labels.to(args.device).long()).item()

            model.train()

            outer_total_loss += intermediate_running_total_loss # The total_loss resets every args.log_interval batches. This is the total loss over the whole epoch / dataset.
            outer_total_correct += intermediate_running_total_correct
            outer_total_datapoints += intermediate_running_total_datapoints
            
            intermediate_running_total_loss = 0
            intermediate_running_total_correct = 0
            intermediate_running_total_datapoints = 0

            train_losses.append(avg_train_loss) # Save the training loss for the last args.log_interval batches
            train_accuracies.append(avg_train_accuracy)

            # Save the validation metrics
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            # Weights and Biases logging
            wandbdict = {f"train_loss (last {logging_interval} batches)": avg_train_loss, f"train_accuracy (last {logging_interval})": avg_train_accuracy, "test_loss": test_loss, "test_accuracy": test_accuracy}
            if args.only_measure_masses:
                for namedparam, nmlsr in namedparam_normaliser_dict.items():
                    wandbdict[f"normaliser/{namedparam[0]}"] = nmlsr
            # wandb.log(wandbdict)

            print(f"Iteration: {iteration}, Train Loss: {avg_train_loss}, Train Accuracy: {avg_train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Time: {time.time() - start_time}", flush=True)

        if args.use_scheduler:
            dummyoptimiseroldlr = dummyoptimiserforscheduler.param_groups[0]['lr']
            dummyscheduler.step()

            ratiobetweenthissteplrandlaststeplrdummy = dummyoptimiserforscheduler.param_groups[0]['lr'] / dummyoptimiseroldlr

            # For each group in the optimiser, decay the learning rate by the same ratio as the scheduler. This is supposed to be equivalent to scheduling the FSLRs.
            # At the start of training, we know the base FSLRs will match the current model's. We assume with scheduling that the baseFSLRs just decay like the schedule times the initial FSLR, because we assume FSLR is proportional to LR.
            # Therefore, since we only use FLeRM at start of training, multiplying the LRs (which are now different because they have been normalised to ensure the FSLRs match the base FSLRs) by the scheduler (equivalently decaying by the same fixed percentage) should be equivalent to scheduling the FSLRs, and so it should match the base model's schedule.
            # In theory
            for param_group in optimiser.param_groups:
                param_group['lr'] *= ratiobetweenthissteplrandlaststeplrdummy
            
            # Every 200 batches, print the learning rates to check they're decaying properly
            if iteration % 200 == 0:
                print(f"Learning rates: {[param_group['lr'] for param_group in optimiser.param_groups]}")
                if not args.only_flerm_first_step_ablation:
                    print("Warning: Using the scheduler expects you to use the args.only_flerm_first_step_ablation flag, because we implemented it based on that assumption. (possible you are just measuring masses, in which case ignore this warning)")

        iteration += 1

    # Compute epoch train loss, test loss, train accuracy, and test accuracy
    y_pred_train = model(in_memory_train_set.images.flatten(1,3))
    _, predicted_train = torch.max(y_pred_train, 1)
    correct_train = (predicted_train == in_memory_train_set.labels).sum().item()
    accuracy_train = correct_train / train_labels.size(0)

    train_loss = loss_fn(y_pred_train.squeeze(), in_memory_train_set.labels.long()).item()

    # y_pred_test = model(in_memory_test_set.images.flatten(1,3))
    # _, predicted_test = torch.max(y_pred_test, 1)
    # correct_test = (predicted_test == in_memory_test_set.labels).sum().item()
    # accuracy_test = correct_test / test_labels.size(0)

    # test_loss = loss_fn(y_pred_test.squeeze(), test_labels.to(args.device).long()).item()

    end_time = time.time()

    print(f'Epoch: {i} complete. Time taken: {end_time - start_time}', flush=True)


# wandb.finish()

# Save the results
results = {'train_losses': train_losses, 'test_losses': test_losses, 'train_accuracies': train_accuracies, 'test_accuracies': test_accuracies}

if args.save_name is not None:
    torch.save(results, args.save_name)
else:
    if args.normalised:
        normstring = "normalised"
    else:
        normstring = "unnormalised"
    nameprefix = args.ptprefix + normstring + args.optimiser + args.model
    torch.save(results, f"{nameprefix}_results_widthmult_{args.width_mult}_depthmult_{args.depth_mult}_lr_{args.lr}_seed_{args.seed}.pt")

if args.only_measure_masses:
    # Save the normalisers_iters_dict
    torch.save(normalisers_iters_dict, f"{args.ptnormsprefix}basemodel{args.model}cifar10empiricalmasses_lr_{args.lr}_seed_{args.seed}.ptnorms")
