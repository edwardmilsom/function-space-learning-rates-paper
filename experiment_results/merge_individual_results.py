import argparse
import glob
import torch

parser = argparse.ArgumentParser(description='Merge individual results')
parser.add_argument("--name_prefix", type=str, default="sgd_results")
parser.add_argument("--allowmissingfiles", action="store_true")
parser.add_argument("--transferparam", type=str, choices=["widthmult", "depthmult","initscale"], default="widthmult")
parser.add_argument("--perplexity_or_accuracy", type=str, choices=["perplexity", "accuracy"], default="perplexity")
parser.add_argument("--default_other_transferparam_value", type=int, default=1)
parser.add_argument("--onlytrainloss", action="store_true")
args = parser.parse_args()

name_prefix = args.name_prefix

# Find all files of the form "{name_prefix}_widthmult_{widthmultiplier}_depthmult_{depthmultiplier}_lr_{lr}_seed_{seed}.pt" (for now. This script is only designed to actually utilise one transfer parameter at a time, either width or depth)
# Maybe an idea would be to assume that the transfer parameters are in alphabetical order, so that we always know how the filename is structured.
files = glob.glob(f"{name_prefix}*_{args.transferparam}_*_lr_*_seed_*.pt")

# filenames should be of the structures "{name_prefix}_OTHERTRANSFERPARAMSBEFORE_args.transferparam_{transferparamval}_OTHERTRANSFERPARAMSAFTER_lr_{lr}_seed_{seed}.pt"
examplefilename = files[0]
transferparam_transferval = args.transferparam + "_" + examplefilename.split(f"_{args.transferparam}_")[1].split("_")[0]
othertransferparamsbefore = examplefilename.split(f"{name_prefix}")[1].split(f"_{args.transferparam}_")[0]
othertransferparamsbefore = othertransferparamsbefore + "_"
othertransferparamsafter = examplefilename.split(transferparam_transferval)[1].split("_lr_")[0]

# Find all unique widths and learning rates
transferparamvals = set()
lrs = set()
seeds = set()
for file in files:
    if args.transferparam == "widthmult" or args.transferparam == "depthmult":
        transferparamval = int(file.split(f"_{args.transferparam}_")[1].split("_")[0])
    elif args.transferparam == "initscale":
        transferparamval = float(file.split(f"_{args.transferparam}_")[1].split("_")[0])
    else:
        raise ValueError("Invalid transfer parameter")
    lr = float(file.split("_lr_")[1].split("_")[0])
    seed = int(file.split("_seed_")[1].split(".pt")[0])
    transferparamvals.add(transferparamval)
    lrs.add(lr)
    seeds.add(seed)

# Sort the widths and learning rates
transferparamvals = sorted(list(transferparamvals))
lrs = sorted(list(lrs))
seeds = sorted(list(seeds))

# Each file "{name_prefix}_width_{width}_lr_{lr}_seed_{seed}.pt" contains a dictionary with the following keys: "train_losses", "test_losses", "train_accuracies", "test_accuracies".
# Each value is a list of metrics per epoch. We will merge these dictionaries into a single dictionary.

# This file should now be more flexible, since it can handle both perplexity and accuracy
if args.perplexity_or_accuracy == "perplexity":
    metrics = ["train_losses", "test_losses"]#, "train_perplexities", "test_perplexities"]
else:
    metrics = ["train_losses", "test_losses", "train_accuracies", "test_accuracies"]

if args.onlytrainloss:
    metrics = ["train_losses"]

def build_filename(transferparamval, lr, seed):
    # if args.transferparam == "widthmult":
    #     return f"{name_prefix}_widthmult_{transferparamval}_depthmult_{args.default_other_transferparam_value}_lr_{lr}_seed_{seed}.pt"
    # elif args.transferparam == "depthmult":
    #     return f"{name_prefix}_widthmult_{args.default_other_transferparam_value}_depthmult_{transferparamval}_lr_{lr}_seed_{seed}.pt"
    # elif args.transferparam == "initscale":
    #     raise NotImplementedError("Initscale not implemented yet")
    return f"{name_prefix}{othertransferparamsbefore}{args.transferparam}_{transferparamval}{othertransferparamsafter}_lr_{lr}_seed_{seed}.pt"
    # else:
        # raise ValueError("Invalid transfer parameter")

# Create the merged dictionary
merged_dict = {}
for key in metrics:
    merged_dict[key] = {}
    for transferparamval in transferparamvals:
        merged_dict[key][transferparamval] = {}
        for lr in lrs:
            merged_dict[key][transferparamval][lr] = {}
            for seed in seeds:
                try:
                    file_dict = torch.load(build_filename(transferparamval, lr, seed))
                    merged_dict[key][transferparamval][lr][seed] = file_dict[key]
                except FileNotFoundError:
                    if not args.allowmissingfiles:
                        raise FileNotFoundError(f"File {build_filename(transferparamval, lr, seed)} not found. Double check your files are as expected before using the --allowmissingfiles flag.")
                    # If the file does not exist, fill in the dictionary with None
                    else:
                        merged_dict[key][transferparamval][lr][seed] = None

from math import nan

# Ensure that all metrics have the same number of transfervals, learning rates, seeds, and epochs (except when they are None)
num_transfervals = None
num_lrs = None
num_logs = {key:None for key in merged_dict.keys()} # Define the number of epochs / iterations / logs for each metric (since test and train might be logged at different intervals)
for key in merged_dict:
    for transferval in merged_dict[key]:
        for lr in merged_dict[key][transferval]:
            for seed in merged_dict[key][transferval][lr]:
                if merged_dict[key][transferval][lr][seed] is not None:
                    if num_transfervals is None:
                        num_transfervals = len(merged_dict[key])
                        num_lrs = len(merged_dict[key][transferval])
                        num_seeds = len(merged_dict[key][transferval][lr])
                        num_logs[key] = len(merged_dict[key][transferval][lr][seed])
                    elif num_logs[key] == None:
                        num_logs[key] = len(merged_dict[key][transferval][lr][seed])
                    else:
                        if len(merged_dict[key][transferval][lr][seed]) < num_logs[key]:
                            print(f"Warning:Empty log. Filling missing entries with NaN (nameprefix={name_prefix}, key={key}, transferval={transferval}, lr={lr}, seed={seed})")
                            for i in range(len(merged_dict[key][transferval][lr][seed]), num_logs[key]):
                                merged_dict[key][transferval][lr][seed].append(nan)
                        assert len(merged_dict[key]) == num_transfervals
                        assert len(merged_dict[key][transferval]) == num_lrs
                        assert len(merged_dict[key][transferval][lr]) == num_seeds
                        assert len(merged_dict[key][transferval][lr][seed]) == num_logs[key]

# Replace all None values with a list of num_epochs NaNs
for key in merged_dict:
    for transferval in merged_dict[key]:
        for lr in merged_dict[key][transferval]:
            for seed in merged_dict[key][transferval][lr]:
                if merged_dict[key][transferval][lr][seed] is None:
                    merged_dict[key][transferval][lr][seed] = [nan for _ in range(num_logs[key])]

# Save the merged dictionary
torch.save(merged_dict, f"{name_prefix}_{args.transferparam}transfer_merged.pt")

