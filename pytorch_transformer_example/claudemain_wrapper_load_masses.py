# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-103 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='Transformer',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
# parser.add_argument('--nlayers', type=int, default=2,
#                     help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--mps', action='store_true', default=False,
                        help='enables macOS GPU training')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--ptprefix', type=str, default='', help='Prefix for the save .pt file at the end of the run')
parser.add_argument('--ptnormsprefix', type=str, default='', help='Prefix for the save .ptnorms file for the empirical masses')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--width_mult', type=int, default=1,
                    help='width multiplier for the transformer model')
parser.add_argument('--depth_mult', type=int, default=1,
                    help='depth multiplier for the transformer model')
parser.add_argument('--optimiser', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--normalised', action='store_true', help='Use normalised update')
parser.add_argument('--normaliser_update_frequency', type=int, default=100)
parser.add_argument('--save_name', type=str, help='Path of the file to save the results to, usually with a .pt extension. E.g. \"yourfilename.pt\"')
parser.add_argument('--normaliser_beta', type=float, default=0.9)
parser.add_argument('--normaliser_approx_type', type=str, default='kronecker', choices=['kronecker', 'iid', 'full_cov'])
parser.add_argument('--transformer_depth_norm', action='store_true', help='Use normalised update for the transformer depth scaling')
parser.add_argument('--measure_masses_only', action='store_true', help='Only measure the masses, do not modify the learning rates')
parser.add_argument('--normtype', type=str, default='postnorm', choices=['postnorm', 'prenorm', "postnormpreres","nonorm"], help='Type of normalisation to use in the transformer')
parser.add_argument('--init_scale', type=float, default=1.0, help='Initial scale for the feedforward layer in the transformer')
parser.add_argument('--noqknorm', action='store_true', help='Use normalisation in the query-key dot product in the transformer')
parser.add_argument('--equal_mass_ablation', action='store_true', help='Use equal masses for all parameters')
parser.add_argument('--only_flerm_first_step_ablation', action='store_true', help='Only use the flerm for the first step of the normaliser update')
parser.add_argument('--equal_mass_but_still_splitting_depth_properly_ablation', action='store_true', help='Use equal masses for all parameters, but still split the masses properly for the depth scaling')
# parser.add_argument('--use_forward_pass_rootL', action='store_true', help='Use the forward pass root L in the transformer')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
import random
import numpy as np
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Don't use the deterministic algorithms because they're slow and often don't work without environment variables set
# # Enable deterministic algorithms
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# Don't disable tf32 computations because they're faster
# # Disable tf32 computations
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

# In fact, let's explicitly enable tf32 computations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
# train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = corpus.vocab_size
from decoder_transformer import DecoderOnlyTransformer
if args.model == 'Transformer':
    if args.transformer_depth_norm:
        # model = model.DepthScalingTransformerModel(ntokens, args.emsize * args.width_mult, args.nhead, args.nhid * args.width_mult, 2 * args.depth_mult, args.dropout, args.depth_mult).to(device)
        # def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, widthmult=1.0, depthmult=1.0)
        model = DecoderOnlyTransformer(vocab_size=ntokens, d_model=args.emsize, num_heads=args.nhead, num_layers=2, d_ff=4*args.emsize, widthmult=args.width_mult, depthmult=args.depth_mult, normtype=args.normtype, init_scale=args.init_scale, noqknorm=args.noqknorm, use_forward_pass_rootL=True).to(device)
    else:
        raise ValueError("DIDN'T EXPECT TO NOT USE DEPTH SCALING")
        model = model.TransformerModel(ntokens, args.emsize * args.width_mult, args.nhead, args.nhid * args.width_mult, 2 * args.depth_mult, args.dropout).to(device)
else:
    raise ValueError("WE SHOULD NOT BE TRYING TO USE AN RNN")
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss(reduction='sum')

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

# I have modified this so that batch dimension is the first dimension, and the sequence length is the second dimension.
def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len].transpose(0,1).contiguous()
    target = source[i+1:i+1+seq_len].transpose(0,1).contiguous().view(-1)
    return data, target


from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, tokens, sequence_length):
        """
        Initialize the dataset with tokens and sequence length
        
        Args:
            tokens (torch.Tensor): 1D tensor of token ids
            sequence_length (int): Length of sequences to generate
        """
        self.tokens = tokens.to(device)
        self.sequence_length = sequence_length
        
        # Calculate number of complete sequences we can make
        # -1 because we need one extra token for each target
        self.n_sequences = (len(tokens) - 1) // sequence_length
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Calculate start position for this sequence
        start_idx = idx * self.sequence_length
        
        # Get sequence
        sequence = self.tokens[start_idx:start_idx + self.sequence_length]
        # Get targets (next token after each position in sequence)
        targets = self.tokens[start_idx + 1:start_idx + self.sequence_length + 1]
        
        return sequence, targets

def create_sequence_dataloader(tokens, batch_size, sequence_length, shuffle=True, num_workers=0):
    """
    Create a DataLoader for sequence prediction
    
    Args:
        tokens (torch.Tensor): Input token tensor
        batch_size (int): Number of sequences per batch
        sequence_length (int): Length of each sequence
        shuffle (bool): Whether to shuffle the data
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = SequenceDataset(tokens, sequence_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

# Create dataloader for training, but just stick with the old code for validation and test because those don't need shuffling
train_loader = create_sequence_dataloader(corpus.train, args.batch_size, args.bptt, shuffle=True)
flerm_train_loader = create_sequence_dataloader(corpus.train, args.batch_size, args.bptt, shuffle=True)
flerm_train_loader_iter = iter(flerm_train_loader)
# val_loader = create_sequence_dataloader(corpus.valid, eval_batch_size, args.bptt)
# test_loader = create_sequence_dataloader(corpus.test, eval_batch_size, args.bptt)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_loss_divisor = 0.
    ntokens = corpus.vocab_size
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += criterion(output, targets).item()
            total_loss_divisor += targets.size(0)
    return total_loss / total_loss_divisor


def train():
    global flerm_train_loader_iter # Python treats this as a local variable because we reassign it in the except block, so just declare it as global
    # Reset cuda peak memory tracker
    torch.cuda.reset_peak_memory_stats()
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    total_loss_divisor = 0.
    outer_total_loss = 0.
    outer_total_loss_divisor = 0
    start_time = time.time()
    ntokens = corpus.vocab_size
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch, dataandtargets in enumerate(train_loader):
        # data, targets = get_batch(train_data, i)
        data, targets = dataandtargets
        data = data.to(device)
        targets = targets.to(device)
        targets = targets.view(-1) # Flatten the targets, because that's what the loss function expects
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if args.model == 'Transformer':
            optimiser.zero_grad()

            if args.normalised and batch % args.normaliser_update_frequency == 0:
                normaliser.save_weights() # Save the weights before the update
                # old_masses = normaliser.masses
                if not args.measure_masses_only and not args.equal_mass_ablation:
                    normaliser.set_masses(generate_masses_dict(batch//args.normaliser_update_frequency, model)) # Set the masses for the current update.
                # print(f"batch//args.normaliser_update_frequency: {batch//args.normaliser_update_frequency}")
                # diff_masses = [old_masses[i] - normaliser.masses[i] for i in range(len(old_masses))]
                # print(f"diff_masses: {diff_masses}")

            output = model(data)
            output = output.view(-1, ntokens)
            
            loss = criterion(output, targets)
            loss.backward() # We do not retain graph here because we favour memory usage over speed. Also retaining graph makes in-place operations on the weights difficult.
            optimiser.step()

            #normaliser.scheduler_step() # Step the scheduler. TODO.
            
            if args.normalised and batch % args.normaliser_update_frequency == 0:
                # Do 40 warmup iterations at the start of training (run the normaliser for a few batches without updating the learning rates)
                # Note: normaliser.update_lrs usually in_place overwrites the tensor created in normaliser.save_weights() to become the updates to the weights, so we need a flag reuse_previous_weight_updates=True to prevent this when doing warmup, as the optimiser step isn't changing.
                if batch == 0:
                    # Run first time to compute weight_updates
                    try:
                        flerm_data, _ = next(flerm_train_loader_iter)
                    except StopIteration:
                        flerm_train_loader_iter = iter(flerm_train_loader)
                        flerm_data, _ = next(flerm_train_loader_iter)
                    normaliser.update_lrs(flerm_data, modify_lrs=False) 
                    for _ in range(39):
                        try:
                            flerm_data, _ = next(flerm_train_loader_iter)
                        except StopIteration:
                            flerm_train_loader_iter = iter(flerm_train_loader)
                            flerm_data, _ = next(flerm_train_loader_iter)
                        normalisers = normaliser.update_lrs(flerm_data, modify_lrs=False, return_delta_ell_fs=True, reuse_previous_weight_updates=True) # Don't try and recompute the weight updates, or it'll break, because you'll lose the information we stored in normaliser.save_weights()
                    # Do the actual normaliser update, but have to use reuse_previous_weight_updates=True because we already ran the normaliser for this optimiser step
                    try:
                        flerm_data, _ = next(flerm_train_loader_iter)
                    except StopIteration:
                        flerm_train_loader_iter = iter(flerm_train_loader)
                        flerm_data, _ = next(flerm_train_loader_iter)
                    # if (flerm_data == data).all():
                    #     print("WARNING: flerm_data is the same as data. This should be extremely rare.")
                    if args.measure_masses_only:
                        normalisers = normaliser.update_lrs(flerm_data, modify_lrs=False, return_delta_ell_fs=True, reuse_previous_weight_updates=True) # Instead of actually changing the LRs, just return what the estimate dF caused by each parameter's update is (for LR=1).
                        namedparam_normaliser_dict = {namedparam: nmlsr for namedparam, nmlsr in zip(model.named_parameters(), normalisers)}
                        # Print name: normaliser for each parameter
                        for namedparam, nmlsr in namedparam_normaliser_dict.items():
                            normalisers_iters_dict[namedparam[0]].append(nmlsr)
                    else:
                        normaliser.update_lrs(flerm_data, reuse_previous_weight_updates=True) # Replace the last update with the normalised update, and update the learning rates for subsequent updates.
                
                # For all other iterations, do the normal things
                elif not args.only_flerm_first_step_ablation:
                    # Do the actual normaliser update
                    try:
                        flerm_data, _ = next(flerm_train_loader_iter)
                    except StopIteration:
                        flerm_train_loader_iter = iter(flerm_train_loader)
                        flerm_data, _ = next(flerm_train_loader_iter)
                    # if (flerm_data == data).all():
                    #     print("WARNING: flerm_data is the same as data. This should be extremely rare.")
                    if args.measure_masses_only:
                        normalisers = normaliser.update_lrs(flerm_data, modify_lrs=False, return_delta_ell_fs=True) # Instead of actually changing the LRs, just return what the estimate dF caused by each parameter's update is (for LR=1).
                        namedparam_normaliser_dict = {namedparam: nmlsr for namedparam, nmlsr in zip(model.named_parameters(), normalisers)}
                        # Print name: normaliser for each parameter
                        for namedparam, nmlsr in namedparam_normaliser_dict.items():
                            normalisers_iters_dict[namedparam[0]].append(nmlsr)
                    else:
                        normaliser.update_lrs(flerm_data) # Replace the last update with the normalised update, and update the learning rates for subsequent updates.
            
        else:
            raise ValueError("WE SHOULD NOT BE TRYING TO USE AN RNN IN THE TRAINING LOOP")
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        total_loss_divisor += data.size(0) * data.size(1)

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / total_loss_divisor
            elapsed = time.time() - start_time
            model.eval()
            val_loss = evaluate(val_data)
            # val_perplexity = math.exp(val_loss)
            model.train()
            # print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
            #         'train loss {:5.2f} | train ppl {:8.2f}'.format(
            #     epoch, batch, len(train_data) // args.bptt, args.lr,
            #     elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)), flush=True)
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'train loss {:5.2f} | train ppl {:8.2f} | val loss {:5.2f} | val ppl {:8.2f}'.format(
                epoch, batch, len(corpus.train) // (args.bptt * args.batch_size), args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), val_loss, math.exp(val_loss)), flush=True)
            print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB", flush=True)
            outer_total_loss += total_loss # The total_loss resets every args.log_interval batches. This is the total loss over the whole epoch / dataset.
            outer_total_loss_divisor += total_loss_divisor
            total_loss = 0
            total_loss_divisor = 0
            start_time = time.time()
            
            train_loss = outer_total_loss / outer_total_loss_divisor
            # train_perplexity = math.exp(train_loss)

            train_losses.append(cur_loss) # Save the training loss for the last args.log_interval batches
            # train_perplexities.append(math.exp(cur_loss))

            # Save the validation metrics
            val_losses.append(val_loss)
            # train_perplexities.append(train_perplexity)
            # val_perplexities.append(val_perplexity)

        if args.dry_run:
            break

    train_loss = outer_total_loss / outer_total_loss_divisor
    # print peak cuda memory
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024**2} MB")
    return train_loss
    

#normalised_optimiser is in the parent directory, so we need to import it from ..
os.sys.path.append("..")
from normalised_optimiser import WrapperUpdateNormaliser

# Create a parameter group for each parameter in the model, so that we can set the learning rate for each parameter individually
if args.normalised and not args.measure_masses_only:
    param_groups = [{'params': [p], 'lr': args.lr} for p in model.parameters()] # If we're using the normalised update, the internal learning rate shouldn't matter, but it does.
else:
    param_groups = [{'params': [p], 'lr': args.lr} for p in model.parameters()]
# param_groups = [{'params': [p], 'lr': args.lr} for p in model.parameters()]

if args.optimiser == 'adam':
    # optimiser = torch.optim.Adam(model.parameters(), lr=1)
    optimiser = torch.optim.Adam(param_groups)
elif args.optimiser == 'sgd':
    # optimiser = torch.optim.SGD(model.parameters(), lr=1)
    optimiser = torch.optim.SGD(param_groups)

# # Initialise the network by running a forward pass
# data, targets = get_batch(train_data, 0)
# output = model(data)

# Find the indices of the input embedding and readout layer weights / biases
input_embedding_mass_idx = None
readout_layer_weight_mass_idx = None
readout_layer_bias_mass_idx = None
idx = 0
for name, param in model.named_parameters():
    if name == 'input_emb.weight':
        input_embedding_mass_idx = idx
    elif name == 'output_proj.weight':
        readout_layer_weight_mass_idx = idx
    elif name == 'output_proj.bias':
        readout_layer_bias_mass_idx = idx

    idx += 1

if input_embedding_mass_idx is None or readout_layer_weight_mass_idx is None or readout_layer_bias_mass_idx is None:
    raise ValueError("Could not find the input embedding or readout layer weights / biases in the model")

if args.equal_mass_ablation and args.equal_mass_but_still_splitting_depth_properly_ablation:
    raise ValueError("Cannot use both --equal_mass_ablation and --equal_mass_but_still_splitting_depth_properly_ablation")

masses = {}

# Fetch observed masses from the training runs, and average them over the seeds
if not args.measure_masses_only and args.normalised and not args.equal_mass_ablation:
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    #seeds = [args.seed]
    obs_masses_avg_seeds_dict_inited = False
    obs_masses_avg_seeds_dict = {}
    if not args.equal_mass_but_still_splitting_depth_properly_ablation:
        for seed in seeds:
            single_seed_observed_masses_training_dict = torch.load(f"{args.ptnormsprefix}{args.normtype}transformerbasemodelempiricalmasses_lr_{args.lr}_seed_{seed}.ptnorms")
            for key in single_seed_observed_masses_training_dict:
                for i in range(len(single_seed_observed_masses_training_dict[key])):
                    single_seed_observed_masses_training_dict[key][i] = single_seed_observed_masses_training_dict[key][i].item() # Convert the tensors to floats
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
        example_mass_file = torch.load("forwardPassRootLpostnormprerestransformerbasemodelempiricalmasses_lr_0.0001165964390619342_seed_0.ptnorms")
        base_param_names = list(example_mass_file.keys())
        obs_masses_avg_seeds_dict = {name:[] for name in base_param_names}

    # Depthmult means more parameters than we have masses for, so we need to divide the masses between the parameters
    def generate_masses_dict(step, model):
        model_par_names = [name for name, _ in model.named_parameters()]
        if args.equal_mass_but_still_splitting_depth_properly_ablation:
            observed_masses_this_step = {name:1/len(obs_masses_avg_seeds_dict.keys()) for name, mass_iters in obs_masses_avg_seeds_dict.items()}
        else:
            observed_masses_this_step = {name:mass_iters[step] for name, mass_iters in obs_masses_avg_seeds_dict.items()}
        observed_masses_names = list(observed_masses_this_step.keys())
        generated_masses = {}
        modelparsstartwith_orig_mod = True
        for name in model_par_names:
            modelparsstartwith_orig_mod = name.startswith("_orig_mod.") and modelparsstartwith_orig_mod
        for name in observed_masses_names:
            if (not name.startswith("_orig_mod.")) and modelparsstartwith_orig_mod:
                name = "_orig_mod." + name
                observedkey = name.replace("_orig_mod.", "")
            elif name.startswith("_orig_mod.") and (not modelparsstartwith_orig_mod):
                name = name.replace("_orig_mod.", "")
                observedkey = "_orig_mod." + name
            else:
                observedkey = name
            if "layers" in name:
                old_layer_num = int(name.split("layers.")[1].split(".")[0])
                new_layer_num = args.depth_mult*old_layer_num
                for d in range(args.depth_mult):
                    param_name = name.replace(f"layers.{old_layer_num}.", f"layers.{new_layer_num+d}.")
                    generated_masses[param_name] = observed_masses_this_step[observedkey] / args.depth_mult
            else:
                generated_masses[name] = observed_masses_this_step[observedkey]
        
        for name in model_par_names:
            assert name in generated_masses, f"Name {name} not in generated masses"
        for name in generated_masses:
            assert name in model_par_names, f"Name {name} not in model par names"
        
        return generated_masses

    masses = generate_masses_dict(0, model)

# Initialise the normaliser
normaliser = WrapperUpdateNormaliser(model, optimiser, outerlr=args.lr, beta=args.normaliser_beta, approx_type=args.normaliser_approx_type, masses = masses)

train_losses = []
val_losses = []
# train_perplexities = []
# val_perplexities = []
# model = torch.compile(model)
# At any point you can hit Ctrl + C to break out of training early.
if args.measure_masses_only:
    normalisers_iters_dict = {name:[] for name, param in model.named_parameters()}
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train()
        # train_perplexity = math.exp(train_loss)
        val_loss = evaluate(val_data)
        val_perplexity = math.exp(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)), flush=True)
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(args.save, 'wb') as f:
        #         torch.save(model, f)
        #     best_val_loss = val_loss
        # else:
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 4.0

        # train_losses.append(train_loss)
        # val_losses.append(val_loss)
        # train_perplexities.append(train_perplexity)
        # val_perplexities.append(val_perplexity)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


# # Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)
#     # after load the rnn params are not a continuous chunk of memory
#     # this makes them a continuous chunk, and will speed up forward pass
#     # Currently, only rnn model supports flatten_parameters function.
#     if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
#         model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# if len(args.onnx_export) > 0:
#     # Export the model in ONNX format.
#     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

# results = {'train_losses': train_losses, 'test_losses': val_losses, 'train_perplexities': train_perplexities, 'test_perplexities': val_perplexities}
results = {'train_losses': train_losses, 'test_losses': val_losses}
# torch.save(results, f"results_widthmult_{args.width_mult}_lr_{args.lr}.pt")
# if args.optimiser == 'adam' and not args.normalised:
#     torch.save(results, f"{args.save_dir}/adam_results_widthmult_{args.width_mult}_lr_{args.lr}.pt")
# elif args.optimiser == 'sgd' and not args.normalised:
#     torch.save(results, f"{args.save_dir}/sgd_results_widthmult_{args.width_mult}_lr_{args.lr}.pt")
# elif args.optimiser == "sgd" and args.normalised:
#     torch.save(results, f"{args.save_dir}/normalisedsgd_results_widthmult_{args.width_mult}_lr_{args.lr}.pt")
# elif args.optimiser == "adam" and args.normalised:
#     torch.save(results, f"{args.save_dir}/normalised_adam_results_widthmult_{args.width_mult}_lr_{args.lr}.pt")

if args.save_name is not None:
    torch.save(results, args.save_name)
else:
    if args.normalised and not args.measure_masses_only:
        normstring = "normalised"
    elif args.normalised and args.measure_masses_only:
        normstring = "measuremasses"
    else:
        normstring = ""
    nameprefix = args.ptprefix + normstring + args.optimiser
    torch.save(results, f"{nameprefix}_results_widthmult_{args.width_mult}_depthmult_{args.depth_mult}_initscale_{args.init_scale}_lr_{args.lr}_seed_{args.seed}.pt")

if args.measure_masses_only:
    # Save the normalisers_iters_dict
    torch.save(normalisers_iters_dict, f"{args.ptnormsprefix}{args.normtype}transformerbasemodelempiricalmasses_lr_{args.lr}_seed_{args.seed}.ptnorms")
