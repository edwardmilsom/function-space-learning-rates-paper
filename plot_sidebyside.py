import torch
from matplotlib import pyplot as plt
import numpy as np
import warnings
import math
import argparse

from transfer_plotting_v2 import create_training_plots_on_axis

parser = argparse.ArgumentParser()
parser.add_argument('unnormalisedfilename', type=str, help='Filename of the unnormalised model')
parser.add_argument('normalisedfilename', type=str, help='Filename of the normalised model')
parser.add_argument('--transferparam', type=str, help='Transfer parameter to plot', default='widthmult')
parser.add_argument('--plotmetric', type=str, choices=['train_losses', 'test_losses', 'train_accuracies', 'test_accuracies', 'train_perplexities', 'test_perplexities'], default='train_losses', help='Metric to plot')
parser.add_argument('--modelname', type=str, help='Name of the model', default='Transformer (PreNormPostMod)')
parser.add_argument('--experiment', type=str, help='Name of the experiment', default='onlyflermfirststep')
parser.add_argument('--yrange', type=str, help='Y range for the plot', default='4.4,6.5')

args = parser.parse_args()

# plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{helvet}  % Load Helvetica font
\renewcommand{\familydefault}{\sfdefault}  % Use sans-serif as default
'''

# Define a 2 x 1 grid

# NOTE: I've changed this to a 1 x 2 grid, so none of the comments will make sense anymore.
# fig, axes = plt.subplots(2, 1, figsize=(6.1, 3.1))
fig, axes = plt.subplots(1, 2, figsize=(9.1, 4.6))

filenames = [args.unnormalisedfilename, args.normalisedfilename]

# Call the function for each subplot
for i, ax in enumerate(axes.flat):
    metrics = create_training_plots_on_axis(ax, filename=filenames[i], transferparam=args.transferparam, plotmetric=args.plotmetric)

# Only use the legend on one plot
axes[0].legend()
axes[0].legend(loc='upper left', prop={'size': 10})

# Set the x and y ranges for each plot.
axes[0].set_xlim(1e-5, 0.1)
# axes[0].set_ylim(4.4, 6.5)
axes[1].set_xlim(1e-5, 0.1)
# axes[1].set_ylim(4.4, 6.5)
axes[0].set_ylim([float(x) for x in args.yrange.split(",")])
axes[1].set_ylim([float(x) for x in args.yrange.split(",")])

# Row labels
# For the first subplot
if args.plotmetric == "train_losses":
    axes[0].set_ylabel(r"\large{Train Loss}")
elif args.plotmetric == "test_losses":
    axes[0].set_ylabel(r"\large{Test Loss}")
axes[1].set_ylabel(r"")
# For the second subplot
# axes[1].set_ylabel(r"\large{Train Loss}")

# Put x axis labels on the bottom row
axes[0].set_xlabel("\large Learning Rate")
axes[1].set_xlabel("\large Learning Rate")

# Put titles "Transformer (PostNormPreRes)"
axes[0].set_title(r"\large Standard")
if args.experiment == "onlyflermfirststep":
    axes[1].set_title(r"\large FLeRM")
elif args.experiment == "regularupdateablation" or args.experiment == "equalmassablation" or args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    axes[1].set_title(r"FLeRM (ablation, see caption)")
# axes[1].set_title(r"\large FLeRM")

# Add a faint background grid to the plots
# for ax in axes.flat:
#     ax.grid(True, linestyle='--', alpha=0.5)

# Add a SECOND y label on the right hand side, "Standard" for the first row and "Mountain" for the second row. Do this by creating a second invisible y axis on the right hand side.
# This is a bit of a hack, but it works.
# Create a second y axis on the right hand side
# ax2 = axes[0].twinx()
# ax2.set_yticks([])
# ax2.set_ylabel(r"\large Standard", rotation=270, labelpad=15)
# ax2.yaxis.set_label_position("right")
ax2lower = axes[1].twinx()
ax2lower.set_yticks([])
ax2lower.set_ylabel(r"Transformer (PreNormPostMod)", rotation=270, labelpad=15)
ax2lower.yaxis.set_label_position("right")

# Set major ticks at every 0.1 on the x axis
# major_ticks = [0.001, 0.01, 0.1, 1, 10]
transformer_major_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
# axes[0].xaxis.set_major_locator(plt.FixedLocator(major_ticks))
# axes[1].xaxis.set_major_locator(plt.FixedLocator(major_ticks))
axes[0].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[1].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))

# Set minor ticks at every ie-j on the x axis
# minor_ticks = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 3, 4, 5, 6, 7, 8, 9]
# axes[0].xaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
# axes[1].xaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
transformer_minor_ticks = [2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1,2e0,3e0,4e0,5e0,6e0,7e0,8e0,9e0,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,2e2,3e2,4e2,5e2,6e2,7e2,8e2,9e2,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3]
axes[0].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[1].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))

plt.tight_layout()
plt.show()
