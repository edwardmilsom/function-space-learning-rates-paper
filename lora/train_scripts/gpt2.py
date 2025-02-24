import torch
import traceback
import random
import math
from transformers import AutoTokenizer, AdamW, AutoModelWithLMHead
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
import numpy as np
import wandb
from pathlib import Path
import pickle as pk
from argparse import Namespace
from dataset import create_pretrain_dataloader
import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Add it to sys.path
sys.path.insert(0, parent_dir)
from flerm import FLeRM
CHECKPOINT = "openai-community/gpt2"

MAX_N_ITER=500
SEQ_LEN = 512

## ---
import fcntl
class Lock:
    def __enter__ (self):
        self.fp = open(LOCKFILE)
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
LOCKFILE="./lock.file"
if not Path(LOCKFILE).exists(): Path(LOCKFILE).touch()

def run_exps(args):
    POSTFIX = f"_gpt2_{args.dataset}_{args.lr_mode}"
    enable_wandb = False
    seeds = [args.seed]
    lrs = [args.lr]
    if args.widthmult is None:
        widthmults = [4, 8, 16, 32]
    else:
        widthmults = [args.widthmult]

    args.normalized = True
    default_args = {'log_interval': 100,
                    'only_measure_masses': args.record_masses,
                    **args.__dict__
                    }

    ## ---------------------- define exps to run
    settings = [
        {
            **default_args,
            'lr': lr,
            'widthmult': widthmult,
            'random_seed': seed,
        }
        for seed in seeds
        for lr in lrs
        for widthmult in widthmults
    ]
    ## ---------------------- define file to store results
    ## ---------------------- if file already exists, load it and adding missing lrs

    def create_widthmult_file(widthmult):
        path = Path(f"./mountain_mathpile_peft_results{POSTFIX}/widthmult{widthmult}_normalized_{args.normalized}_record_masses_{args.record_masses}.pk")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        with Lock():
            with path.open('rb') as f:
                try:
                    results = pk.load(f)
                    # add new lrs
                    for lr in lrs:
                        for widthmult in widthmults:
                            if f"lr={lr}" not in results[f"widthmult={widthmult}"]:
                                results[f"widthmult={widthmult}"][f"lr={lr}"] = {f"seed={seed}": None for seed in seeds}
                except:
                    results = {
                        f"widthmult={widthmult}": {
                            f"lr={lr}": {
                                f"seed={seed}": None
                                for seed in seeds
                            }
                            for lr in lrs
                        }
                        for widthmult in widthmults
                    } # { widthmult -> { lr -> { seed -> metrics } } }
            with path.open('wb') as f:
                pk.dump(results, f)
        return path

    ## ---------------------- run all experiments
    ## ---------------------- updating results file as we go
    ## ---------------------- also tear up/down wandb runs
    for ss in settings:
        path = create_widthmult_file(ss['widthmult'])
        print("GOING TO SAVE TO ", str(path))
        args = Namespace(**ss)
        lr = str(args.lr); seed = args.random_seed; widthmult = args.widthmult
        with Lock():
            with path.open('rb') as f:
                results = pk.load(f)
            rs =results[f"widthmult={widthmult}"][f"lr={lr}"]
            if f"seed={seed}" in rs and rs[f"seed={seed}"] is not None:
                _vseed = results[f"widthmult={widthmult}"][f"lr={lr}"][f"seed={seed}"]
                if args.force:
                    pass
                elif any([math.isnan(x[-1]) for x in _vseed['avg_train_loss']]):
                    print("DETECTED NAN")
                    print("try again")
                elif _vseed['avg_train_loss'][-1][0] <= MAX_N_ITER-200:
                    print("recompute, because not enough iterations")
                    pass
                elif _vseed['avg_train_loss'][-1][-1] <= 0.:
                    print("recompute, because train loss was <= 0.")
                    pass
                elif args.lr > 1e0:
                    print(f"lr too big: {lr}, skipping")
                    continue
                else:
                    print(f"already done! widthmult={widthmult}, lr={lr}, seed={seed}")
                    print("final avg_train_loss", _vseed['avg_train_loss'][-1])
                    continue
        runname = f"width{widthmult}_seed{seed}_lr{args.lr}"
        if enable_wandb:
            wandb.init(project="gpt2_peft_mathpile_proper", name=runname, config=ss, reinit=True)

        metrics = ft_on_mathpile(args, enable_wandb=enable_wandb, POSTFIX=POSTFIX)
        # try:
        #     metrics = ft_on_mathpile(args, enable_wandb=enable_wandb, POSTFIX=POSTFIX)
        # except Exception as e:
        #     metrics = None
        #     from warnings import warn
        #     warn("failed for some reason")
        #     print("=== exc ===")
        #     print(e)
        #     print("=== args===")
        #     print(args)
        #     print("=== continuing anyway===")
            # raise e

        # results[f"widthmult={args.widthmult}"][f"lr={lr}"][f"seed={seed}"] = metrics
        # assert len(metrics['train_loss']) > 0
        if enable_wandb:
            wandb.finish()

        with Lock():
            try:
                with path.open('rb') as f:
                    results = pk.load(f)
                    results[f"widthmult={widthmult}"][f"lr={lr}"][f"seed={seed}"] = metrics
                with path.open('wb') as f:
                    pk.dump(results, f)
            except Exception as e:
                print(traceback.format_exc())
                print("Failed to save results:", e)
                raise e

def ft_on_mathpile(args, enable_wandb, POSTFIX=None):
    print("ARGS")
    print(args)
    measure_masses_path = f"empirical_masses{POSTFIX}/gpt2lora_widthmult{args.widthmult}_lr_{args.lr}_{args.dataset}_seed_{args.random_seed}.ptnorms"
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
    set_seed(args.random_seed)

    batch_size = 8
    num_epochs = 1
    # train_data = FT_Dataset("../LoRA/examples/NLG/data/mathpile/train/train.jsonl", batch_size=batch_size, max_seq_length=512, joint_lm=obj == 'jlm')
    # train_dataloader = DataLoader(
    #     train_data, batch_size=batch_size, num_workers=0,
    #     shuffle=True, pin_memory=False, drop_last=True,
    # )
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataloader = create_pretrain_dataloader(
        tokenizer=tokenizer,
        batch_size=batch_size,
        dataset_name=args.dataset,
        SEQ_LEN=SEQ_LEN
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    model = AutoModelWithLMHead.from_pretrained(CHECKPOINT)
    model.config.pad_token_id = tokenizer.eos_token
    lora_rank = args.widthmult*1
    if args.lora_init == 'pissa':
        lora_config = LoraConfig(r=lora_rank, init_lora_weights='pissa')
    elif args.lora_init == 'gaussian':
        lora_config = LoraConfig(r=lora_rank, init_lora_weights='gaussian')
    else:
        lora_config = LoraConfig(r=lora_rank)
    peft_model = get_peft_model(model, lora_config)
    # print(peft_model)

    # setup parameter groups
    function_space_param_groups = [] # use flerm/mountain to optimize these
    fs_named_parameters = []
    for name, param in peft_model.named_parameters():
        if args.lr_mode == 'fix_a':
            if 'lora_B' in name:
                function_space_param_groups.append({'params': [param], 'lr': args.lr, 'use_flerm': True, 'name': name})
                fs_named_parameters.append((name, param))
            elif 'lora_A' in name:
                function_space_param_groups.append({'params': [param], 'lr': 1e-4, 'use_flerm': True, 'name': name})
                fs_named_parameters.append((name, param))
        elif args.lr_mode == 'fix_b':
            if 'lora_B' in name:
                function_space_param_groups.append({'params': [param], 'lr': 1e-3, 'use_flerm': True, 'name': name})
                fs_named_parameters.append((name, param))
            elif 'lora_A' in name:
                function_space_param_groups.append({'params': [param], 'lr': args.lr, 'use_flerm': True, 'name': name})
                fs_named_parameters.append((name, param))
    fs_optimizer = AdamW(function_space_param_groups) # This is actually Adam, since the default weight decay is 0 in the transformers library.

    if args.joint:
        ## record parameters which are joint in terms of functional learning rate
        joint_parameters = {} # param_name -> [param_name]
        ## here the joint_parameters dict gives us a list of _all other_ 'joint' parameters
        _all_param_names = [name for name, _ in peft_model.named_parameters()]
        for name, module in model.named_modules():
            # Check if module has both LoRA A and B attributes
            has_lora_a = hasattr(module, 'lora_A')
            has_lora_b = hasattr(module, 'lora_B')
            if has_lora_a and has_lora_b:
                lA = f"base_model.model.{name}.lora_A.default.weight"
                lB = f"base_model.model.{name}.lora_B.default.weight"
                assert lA in _all_param_names
                assert lB in _all_param_names
                joint_parameters[lA] = [lB]
                joint_parameters[lB] = [lA]
    else:
        joint_parameters = None

    peft_model.to(device)

    if args.only_measure_masses or not args.normalized:
        masses = {}
    elif args.normalized:
        # Fetch observed masses from the training runs, and average them over the seeds
        obs_masses_avg_seeds_dict_inited = False
        obs_masses_avg_seeds_dict = {} # : param_name -> batch_ix -> masses!
        ## we measure every ~ 100 batches
        # measured_seeds = [0, 1, 2, 3]
        measured_seeds = [1]
        for seed in measured_seeds:
            # single_seed_observed_masses_training_dict = torch.load(f"{args.ptnormsprefix}basemodel{args.model}cifar10empiricalmasses_seed_{seed}.ptnorms")
            # single_seed_observed_masses_training_dict = torch.load(f"empirical_masses/gpt2lora_basemodel_cifar10empiricalmasses_seed_{args.random_seed}.ptnorms")

            load_masses_path = f"empirical_masses{POSTFIX}/gpt2lora_widthmult{args.base_width}_lr_{args.lr}_{args.dataset}_seed_{seed}.ptnorms"
            single_seed_observed_masses_training_dict = torch.load(load_masses_path, weights_only=True)
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
                obs_masses_avg_seeds_dict[key][i] /= len(measured_seeds)

        def generate_masses_dict(step):
            d =  {name:mass_iters[step] for name, mass_iters in obs_masses_avg_seeds_dict.items() if 'lora_' in name}
            if any([math.isnan(x) for x in d.values()]):
                raise ValueError("NAN detected in `generate_masses_dict`")
            return d

        masses = generate_masses_dict(0) # test
    else:
        raise NotImplementedError("Didn't expect to get here, please debug")

    class AverageMeter:
        def __init__(self):
            self.reset()
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    avgmeter = AverageMeter()


    metrics = {'train_loss': [], 'avg_train_loss': [], 'ntokens': []}
    ntokens = 0
    peft_model.train()

    ## ~10k --> 100

    def peft_model_output_closure(inputtuple):
        inputtokens = inputtuple[0]
        msk = inputtuple[1]
        return peft_model(inputtokens, attention_mask=msk).logits


    normalizer = FLeRM(peft_model_output_closure, fs_optimizer,
                                         outerlr=args.lr,
                                         named_parameters=fs_named_parameters,
                                         beta=0.9,
                                         approx_type='kronecker',
                                         baseFSLRs = masses
                                         )
    if args.only_measure_masses:
        normalisers_iters_dict = {name: [] for name, _ in fs_named_parameters}

    init_flerm_n = args.init_flerm_n
    # schedule = [init_flerm_n + 1] + list(range(50, MAX_N_ITER+1, 50))
    schedule = [init_flerm_n + 1] # Only apply FLeRM once, like in the rest of the experiments
    print(schedule)
    for epoch in range(num_epochs):
        iterator_ = range(MAX_N_ITER+1)
        progress_bar = tqdm(range(MAX_N_ITER+1))

        # n_batches = len(train_dataloader)
        n_batches = MAX_N_ITER + 1
        for ix in iterator_:
            batch = next(train_dataloader)
            iter_n = epoch*n_batches + ix
            fs_optimizer.zero_grad()

            if args.normalized and iter_n in schedule or iter_n == init_flerm_n:
                normalizer.save_weights() # Save the weights before the update
                if not args.only_measure_masses and iter_n != init_flerm_n:
                    ix = schedule.index(iter_n)
                    normalizer.set_masses(generate_masses_dict(ix))

            _input = batch['input'].to(device)
            _target = batch['target'].to(device)
            _msk = batch['mask'].to(device)
            Bi, Ti = _input.size()
            Bt, Tt = _target.size()
            ntokens += Ti + Tt
            outputs = peft_model(_input, labels=_target, attention_mask=_msk)
            loss = outputs.loss
            avgmeter.update(loss.item())
            loss.backward()
            fs_optimizer.step()

            if args.normalized and iter_n == init_flerm_n:
                ## do 40 lots of warmup
                flerm_batch = next(train_dataloader)
                _flerminput = flerm_batch['input'].to(device)
                _flermmsk = flerm_batch['mask'].to(device)
                normalizer.update_lrs((_flerminput,_flermmsk), modify_lrs=False, reuse_previous_weight_updates=False)
                for _ in range(39):
                    normalisers = normalizer.update_lrs((_flerminput,_flermmsk), modify_lrs=False, reuse_previous_weight_updates=True, return_delta_ell_fs=True)
            elif args.normalized and iter_n in schedule:
                if not args.only_measure_masses:
                    normalizer.update_lrs((_flerminput,_flermmsk)) # Replace the last update with the normalised update, and update the learning rates for subsequent updates.
                elif args.normalized and args.only_measure_masses:
                    normalisers: list = normalizer.update_lrs((_flerminput,_flermmsk), modify_lrs=False, return_delta_ell_fs=True) # Instead of actually changing the LRs, just return what the estimate dF caused by each parameter's update is (for LR=1).
                    for (name,_), nmlsr in zip(fs_named_parameters, normalisers):
                        normalisers_iters_dict[name].append(nmlsr) # param_name -> float

            fs_optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_description(f"tr_loss: {loss.item():.4f}|avg_tr_loss: {avgmeter.avg:.4f}")

            if iter_n % args.log_interval == 0:
                metrics['train_loss'].append((iter_n, loss.item()))
                metrics['avg_train_loss'].append((iter_n, avgmeter.avg))
                metrics['ntokens'].append((iter_n, ntokens))
                if enable_wandb:
                    wandb.log({'train_loss': loss.item(), 'avg_tr_loss': avgmeter.avg, 'ntokens': ntokens, 'step': iter_n}, step=iter_n)
                avgmeter.reset()
                print("resetting metrics")
        progress_bar.refresh()
        progress_bar.close()

        ## save results and the masses
        with Lock():
            alt_metrics = {'train_losses': metrics['avg_train_loss']}

            if hasattr(args, 'save_name') and args.save_name is not None:
                torch.save(alt_metrics, args.save_name)
            else:
                normstring = "normalized" if args.normalized else "unnormalized"
                recorded_masses = "recorded_masses" if args.only_measure_masses else "no_recorded_masses"

                nameprefix = normstring + "_" + recorded_masses
                path = f"alt_mountain_metrics{POSTFIX}/{nameprefix}_{args.dataset}_results_widthmult_{args.widthmult}_depthmult_1_initscale_1.0_lr_{args.lr}_seed_{args.random_seed}.pt"
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(alt_metrics, path)

            if args.only_measure_masses:
                # Save the normalisers_iters_dict
                Path(measure_masses_path).parent.mkdir(parents=True, exist_ok=True)
                try:
                    torch.save(normalisers_iters_dict, measure_masses_path)
                except:
                    torch.save(normalisers_iters_dict, measure_masses_path)
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--widthmult", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=-1.)
    parser.add_argument("--normalized", action='store_true')
    parser.add_argument("--force", action='store_true')
    parser.add_argument("--joint", action='store_true')
    parser.add_argument("--lr_mode", type=str, default='fix_a', choices=['fix_b', 'fix_a'])
    parser.add_argument("--record_masses", action='store_true')
    parser.add_argument("--base_width", default=2, type=int)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--init_flerm_n", type=int, default=5)
    parser.add_argument("--lora_init", type=str, default='gaussian', choices=['default', 'pissa', 'gaussian'])
    parser.add_argument("--dataset", type=str, default='mathpile', choices=['cold_french_law', 'mathpile'])
    # parser.add_argument("--eps", type=float, default=0.)
    args = parser.parse_args()
    if args.lr <= 0.: args.lr = None
    if args.widthmult <= 0: args.widthmult = None
    if args.seed < 0: args.seed = None
    run_exps(args)