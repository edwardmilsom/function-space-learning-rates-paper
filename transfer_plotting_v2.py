import torch
from matplotlib import pyplot as plt
import numpy as np
import warnings
import math

def create_training_plots_on_axis(ax, filename, early_stopping="none", max_iters=None, 
                                  transferparam="widthmult", plotmetric="train_losses", perplexity_or_accuracy="perplexity", 
                                  do_not_ignore_nan=False, average_over_last_n=1, y_axis_logscale=False):
    """
    Creates a plot on the given Axes or a new figure if Axes is not provided.
    
    Args:
        ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
        Other arguments are as in the original function.

    Returns:
        dictionary: best metrics
    """

    warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
    
    merged_dict = torch.load(filename)
    
    if perplexity_or_accuracy == "perplexity":
        metrics = ["train_losses", "test_losses"]
        early_stop_arg_metric_correspondence = {
            "train_loss": "train_losses", 
            "test_loss": "test_losses"
        }
    else:
        metrics = ["train_losses", "test_losses", "train_accuracies", "test_accuracies"]
        early_stop_arg_metric_correspondence = {
            "train_loss": "train_losses", 
            "test_loss": "test_losses",
            "train_accuracy": "train_accuracies", 
            "test_accuracy": "test_accuracies"
        }

    # Calculate iterations for each metric
    num_iters = {}
    for key in metrics:
        transfervals = list(merged_dict[key].keys())
        num_transfervals = len(transfervals)
        lrs = list(merged_dict[key][transfervals[0]].keys())
        num_lrs = len(lrs)
        seeds = list(merged_dict[key][transfervals[0]][lrs[0]].keys())
        num_seeds = len(seeds)
        iters = list(merged_dict[key][transfervals[0]][lrs[0]][seeds[0]])
        num_iters[key] = len(iters)

    # Verify consistency
    for metric in metrics:
        assert len(merged_dict[metric]) == num_transfervals
        for transferval in merged_dict[metric]:
            assert len(merged_dict[metric][transferval]) == num_lrs
            for lr in merged_dict[metric][transferval]:
                assert len(merged_dict[metric][transferval][lr]) == num_seeds
                for seed in merged_dict[metric][transferval][lr]:
                    assert len(merged_dict[metric][transferval][lr][seed]) == num_iters[metric]

    max_iters = {key: max_iters if max_iters is not None else num_iters[key] for key in metrics}

    # Handle early stopping
    if early_stopping != "none":
        if not all(num_iters[metric] == num_iters[metrics[0]] for metric in metrics):
            raise ValueError("Early stopping requires all metrics to have the same number of iterations")
        
        plotting_dict = {}
        for metric in metrics:
            plotting_dict[metric] = {}
            for transferval in merged_dict[metric]:
                plotting_dict[metric][transferval] = {}
                for lr in merged_dict[metric][transferval]:
                    plotting_dict[metric][transferval][lr] = {}
                    for seed in merged_dict[metric][transferval][lr]:
                        earlystopmetric_losses = merged_dict[early_stop_arg_metric_correspondence[early_stopping]][transferval][lr][seed][:max_iters[metric]]
                        min_earlystopmetric_loss = min(earlystopmetric_losses)
                        min_earlystopmetric_loss_epoch = earlystopmetric_losses.index(min_earlystopmetric_loss)
                        
                        val = merged_dict[metric][transferval][lr][seed][min_earlystopmetric_loss_epoch]
                        for d in range(1, average_over_last_n):
                            val += merged_dict[metric][transferval][lr][seed][min_earlystopmetric_loss_epoch - d]
                        plotting_dict[metric][transferval][lr][seed] = val / average_over_last_n
    else:
        # Use last epoch
        plotting_dict = {}
        for metric in metrics:
            plotting_dict[metric] = {}
            for transferval in merged_dict[metric]:
                plotting_dict[metric][transferval] = {}
                for lr in merged_dict[metric][transferval]:
                    plotting_dict[metric][transferval][lr] = {}
                    for seed in merged_dict[metric][transferval][lr]:
                        val = merged_dict[metric][transferval][lr][seed][max_iters[metric]-1]
                        for d in range(1, average_over_last_n):
                            val += merged_dict[metric][transferval][lr][seed][max_iters[metric]-1-d]
                        plotting_dict[metric][transferval][lr][seed] = val / average_over_last_n

    transferparamdisplaynamedict = {
        "widthmult": "Width $\\times$",
        "depthmult": "Depth $\\times$",
        "initscale": "Init. $\\times$"
    }
    
    if perplexity_or_accuracy == "perplexity":
        metric_displaynamedict = {
            "train_losses": "Training Loss",
            "test_losses": "Test Loss"
        }
    else:
        metric_displaynamedict = {
            "train_losses": "Training Loss",
            "test_losses": "Test Loss",
            "train_accuracies": "Training Accuracy",
            "test_accuracies": "Test Accuracy"
        }

    best_metrics = {}
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None  # Indicate that no new figure is created

    # ax.set_xlabel("Learning Rate")
    ax.set_ylabel(metric_displaynamedict[plotmetric])
    ax.set_xscale("log")
    if y_axis_logscale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(-1, 10)

    for transferval in plotting_dict[plotmetric]:
        # Perform plotting on `ax`
        sorted_lrs = sorted(list(plotting_dict[plotmetric][transferval].keys()))
        sorted_metric_all_seeds = [list(plotting_dict[plotmetric][transferval][lr].values()) for lr in sorted_lrs]
        lrs_rows_seeds_columns = np.array(sorted_metric_all_seeds)

        if do_not_ignore_nan:
            sorted_metric_means = np.mean(lrs_rows_seeds_columns, axis=1)
            sorted_metric_stds = np.std(lrs_rows_seeds_columns, axis=1)
        else:
            sorted_metric_means = np.nanmean(lrs_rows_seeds_columns, axis=1)
            sorted_metric_stds = np.nanstd(lrs_rows_seeds_columns, axis=1)

        # Use a progressive colour scheme for the lines, where a larger value in the legend corresponds to a darker colour.
        # Set the colour scheme
        if transferparam == "widthmult":
            cmap = plt.get_cmap('viridis')
        elif transferparam == "depthmult":
            cmap = plt.get_cmap('plasma')
        elif transferparam == "initscale":
            cmap = plt.get_cmap('cividis')
        # Get the colour
        color = cmap((math.log(float(transferval)) - math.log(min(plotting_dict[plotmetric].keys()))) / (math.log(max(plotting_dict[plotmetric].keys())) - math.log(min(plotting_dict[plotmetric].keys()))))
        if transferparam == "initscale":
            # Change transferval to a string which is 4^exponent (in latex). They are all powers of 4, so make sure the exponent is an integer.
            exponent = int(math.log(float(transferval), 4))
            transferval = f"$4^{{{exponent}}}$"
        elif transferparam == "widthmult" or transferparam == "depthmult":
            # Change transferval to a string which is 2^exponent (in latex). They are all powers of 2, so make sure the exponent is an integer.
            exponent = int(math.log(float(transferval), 2))
            transferval = f"$2^{{{exponent}}}$"
        ax.plot(sorted_lrs, sorted_metric_means, 
                label=f"{transferparamdisplaynamedict[transferparam]} {transferval}",
                linewidth=2, color=color)
        ax.fill_between(sorted_lrs, sorted_metric_means - sorted_metric_stds,
                        sorted_metric_means + sorted_metric_stds, alpha=0.1, color=color)

    return best_metrics