import argparse
from transfer_plotting_v2 import create_training_plots_on_axis  # Assuming previous code is in plot_generation.py
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Plot dictionary of hyperparameters and transformations')
    parser.add_argument("filename", type=str, help="Filename of the merged dictionary")
    parser.add_argument("--early_stopping", type=str, 
                       choices=["train_loss", "test_loss", "train_perplexity", "test_perplexity", "none"], 
                       default="none", help="Plot the early stopping metrics")
    parser.add_argument("--max_iters", type=int, default=None, help="Maximum number of epochs")
    parser.add_argument("--transferparam", type=str, 
                       choices=["widthmult", "depthmult", "initscale"], 
                       default="widthmult", help="Transfer parameter to plot")
    parser.add_argument("--perplexity_or_accuracy", type=str, 
                       choices=["perplexity", "accuracy"], 
                       default="perplexity", help="Plot perplexity or accuracy")
    parser.add_argument("--do_not_ignore_nan", action="store_true", help="Do not ignore NaN values")
    parser.add_argument("--average_over_last_n", type=int, default=1, help="Average over the last n values")
    parser.add_argument("--y_axis_logscale", action="store_true", help="Use log scale for y axis")
    parser.add_argument("--plotmetric", type=str, choices=["train_losses", "test_losses", "train_accuracies", "test_accuracies","train_perplexities","test_perplexities"], default="train_losses", help="Metric to plot")

    args = parser.parse_args()
    
    fig, ax = plt.subplots(1, 1)

    metrics = create_training_plots_on_axis(
        ax=ax,
        filename=args.filename,
        early_stopping=args.early_stopping,
        max_iters=args.max_iters,
        transferparam=args.transferparam,
        plotmetric=args.plotmetric,
        perplexity_or_accuracy=args.perplexity_or_accuracy,
        do_not_ignore_nan=args.do_not_ignore_nan,
        average_over_last_n=args.average_over_last_n,
        y_axis_logscale=args.y_axis_logscale
    )

    # Print metrics
    for metric, values in metrics.items():
        for transferval, result in values.items():
            if result['value'] is not None:
                print(f"Best {metric} for {transferval}: {result['value']} (lr={result['lr']})")
    
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
