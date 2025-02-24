import torch
from matplotlib import pyplot as plt
import numpy as np
import warnings
import math
import argparse

from transfer_plotting_v2 import create_training_plots_on_axis

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='Experiment name', default="fixa", choices=["fixa", "fixb"])

args = parser.parse_args()

# plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{helvet}  % Load Helvetica font
\renewcommand{\familydefault}{\sfdefault}  % Use sans-serif as default
'''

#####################
#WIDTH TRANSFER PLOT#
#####################

# Define a grid
fig, axes = plt.subplots(2, 4, figsize=(12, 5))

if args.experiment == "fixa":
    filenames = ["lorafeb16_mathpile_fixa_gpt2_unnormed.pt", "lorafeb16_coldfrenchlaw_fixa_gpt2_unnormed.pt", "lorafeb16_mathpile_fixa_llama3_unnormed.pt", "lorafeb16_coldfrenchlaw_fixa_llama3_unnormed.pt", "lorafeb16_mathpile_fixa_gpt2_normed.pt", "lorafeb16_coldfrenchlaw_fixa_gpt2_normed.pt", "lorafeb16_mathpile_fixa_llama3_normed.pt", "lorafeb16_coldfrenchlaw_fixa_llama3_normed.pt"]
elif args.experiment == "fixb":
    filenames = ["lorafeb16_mathpile_fixb_gpt2_unnormed.pt", "lorafeb16_coldfrenchlaw_fixb_gpt2_unnormed.pt", "lorafeb16_mathpile_fixb_llama3_unnormed.pt", "lorafeb16_coldfrenchlaw_fixb_llama3_unnormed.pt", "lorafeb16_mathpile_fixb_gpt2_normed.pt", "lorafeb16_coldfrenchlaw_fixb_gpt2_normed.pt", "lorafeb16_mathpile_fixb_llama3_normed.pt", "lorafeb16_coldfrenchlaw_fixb_llama3_normed.pt"]
# Call the function for each subplot
for i, ax in enumerate(axes.flat):
    metrics = create_training_plots_on_axis(ax, filename=filenames[i], transferparam="lorarank", onlytrainloss=True, tupleinsteadofvalue=True, average_over_last_n=2)

# Only use the legend on one plot
axes[0, 0].legend()
axes[0, 0].legend(loc='upper left', prop={'size': 8})

# Set the x and y ranges for each plot.
if args.experiment == "fixa":
    axes[0, 0].set_xlim(1e-5, 0.25)
    axes[0, 0].set_ylim(2.2, 3.4)
    axes[0, 1].set_xlim(1e-5, 0.25)
    axes[0, 1].set_ylim(2.8, 4.0)
    axes[0, 2].set_xlim(3e-7, 8e-4)
    axes[0, 2].set_ylim(1.48, 1.52)
    axes[1, 0].set_xlim(1e-5, 0.25)
    axes[1, 0].set_ylim(2.2, 3.4)
    axes[1, 1].set_xlim(1e-5, 0.25)
    axes[1, 1].set_ylim(2.8, 4.0)
    axes[1, 2].set_xlim(3e-7, 8e-4)
    axes[1, 2].set_ylim(1.48, 1.52)
    axes[0, 3].set_xlim(1e-6, 0.05)
    axes[0, 3].set_ylim(1.5, 2.25)
    axes[1, 3].set_xlim(1e-6, 0.05)
    axes[1, 3].set_ylim(1.5, 2.25)
elif args.experiment == "fixb":
    axes[0, 0].set_xlim(1.5e-5, 0.25)
    axes[0, 0].set_ylim(2.4, 3.0)
    axes[0, 1].set_xlim(1.5e-5, 0.25)
    axes[0, 1].set_ylim(2.8, 4.0)
    axes[0, 2].set_xlim(3e-7, 5e-2)
    axes[0, 2].set_ylim(1.48, 1.52)
    axes[1, 0].set_xlim(1.5e-5, 0.25)
    axes[1, 0].set_ylim(2.4, 3.0)
    axes[1, 1].set_xlim(1.5e-5, 0.25)
    axes[1, 1].set_ylim(2.8, 4.0)
    axes[1, 2].set_xlim(3e-7, 5e-2)
    axes[1, 2].set_ylim(1.48, 1.52)
    axes[0, 3].set_xlim(1e-6, 0.05)
    axes[0, 3].set_ylim(1.5, 2.25)
    axes[1, 3].set_xlim(1e-6, 0.05)
    axes[1, 3].set_ylim(1.5, 2.25)

# Row labels
# For the first subplot
axes[0, 0].set_ylabel(r"\large{Train Loss}")
# For the second subplot
axes[1, 0].set_ylabel(r"\large{Train Loss}")

# Remove the y axis labels for the right plots
for i in range(1, 4):
    axes[0, i].set_ylabel("")
    axes[1, i].set_ylabel("")
# Put x axis labels on the bottom row
for i in range(4):
    axes[1, i].set_xlabel("\large Learning Rate")

# Put titles "MLP", "ResMLP" and "Transformer (PostNorm)" and "Transformer (PreNorm)" on the top row
axes[0, 0].set_title(r"\large MathPile, GPT2")
axes[0, 1].set_title(r"\large French, GPT2")
axes[0, 2].set_title(r"\large MathPile, Llama-3.2-1B")
axes[0, 3].set_title(r"\large French, Llama-3.2-1B")

# Add a faint background grid to the plots
# for ax in axes.flat:
#     ax.grid(True, linestyle='--', alpha=0.5)


# Add a SECOND y label on the right hand side, "Standard" for the first row and "Mountain" for the second row. Do this by creating a second invisible y axis on the right hand side.
# This is a bit of a hack, but it works.
# For each row
# Create a second y axis on the right hand side
ax2 = axes[0, -1].twinx()
ax2.set_yticks([])
ax2.set_ylabel(r"\large Standard", rotation=270, labelpad=15)
ax2.yaxis.set_label_position("right")
ax2lower = axes[1, -1].twinx()
ax2lower.set_yticks([])
ax2lower.set_ylabel(r"\large FLeRM", rotation=270, labelpad=15)
# if args.experiment == "onlyflermfirststep":
#     ax2lower.set_ylabel(r"\large FLeRM", rotation=270, labelpad=15)
# elif args.experiment == "regularupdateablation" or args.experiment == "equalmassablation" or args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
#     ax2lower.set_ylabel(r"FLeRM (ablation, see caption)", rotation=270, labelpad=15)
ax2lower.yaxis.set_label_position("right")

# Set major ticks at every 0.1 on the x axis
majortickslist = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
axes[0, 0].xaxis.set_major_locator(plt.FixedLocator(majortickslist))
axes[0, 1].xaxis.set_major_locator(plt.FixedLocator(majortickslist))
axes[0, 2].xaxis.set_major_locator(plt.FixedLocator(majortickslist))
axes[0, 3].xaxis.set_major_locator(plt.FixedLocator(majortickslist))
axes[1, 0].xaxis.set_major_locator(plt.FixedLocator(majortickslist))
axes[1, 1].xaxis.set_major_locator(plt.FixedLocator(majortickslist))
axes[1, 2].xaxis.set_major_locator(plt.FixedLocator(majortickslist))
axes[1, 3].xaxis.set_major_locator(plt.FixedLocator(majortickslist))

# Set minor ticks at every ie-j on the x axis from 1e-7 to 1e-0
minortickslist = [2e-7, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7, 8e-7, 9e-7, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1]
axes[0, 0].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))
axes[0, 1].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))
axes[0, 2].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))
axes[0, 3].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))
axes[1, 0].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))
axes[1, 1].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))
axes[1, 2].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))
axes[1, 3].xaxis.set_minor_locator(plt.FixedLocator(minortickslist))


plt.tight_layout()
plt.show()
