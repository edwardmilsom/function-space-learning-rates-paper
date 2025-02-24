import torch
from matplotlib import pyplot as plt
import numpy as np
import warnings
import math
import argparse

from transfer_plotting_v2 import create_training_plots_on_axis

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, help='Experiment name', default="onlyflermfirststep")

args = parser.parse_args()

# plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{helvet}  % Load Helvetica font
\renewcommand{\familydefault}{\sfdefault}  % Use sans-serif as default
'''

#####################
#DEPTH TRANSFER PLOT#
#####################

# Define a grid
fig, axes = plt.subplots(2, 4, figsize=(12, 5))

# regularupdateablation runs
if args.experiment == "onlyflermfirststep":
    filenames   = filenames   = ["jan24emawarmup_resmlp_unnormed_depth.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_depth.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_depth.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_depth.pt", "jan24emawarmup_resmlp_normed_depth.pt", "jan24emawarmuppostnormqknorm_transformer_normed_depth.pt", "jan24emawarmupprenormqknorm_transformer_normed_depth.pt", "jan24emawarmuppostnormpreresqknorm_transformer_normed_depth.pt"]
# Equal mass ablation (uses same unnormed files as regularupdateablation runs)
elif args.experiment == "equalmassablation":
    filenames = ["jan24emawarmup_resmlp_unnormed_depth.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_depth.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_depth.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_depth.pt", "jan28equalmassablationonlyflermfirststep_resmlp_normed_depth.pt", "jan28equalmassablationonlyflermfirststeppostnormqknorm_transformer_normed_depth.pt", "jan28equalmassablationonlyflermfirststepprenormqknorm_transformer_normed_depth.pt", "jan28equalmassablationonlyflermfirststeppostnormpreresqknorm_transformer_normed_depth.pt"]
elif args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    filenames = ["jan24emawarmup_resmlp_unnormed_depth.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_depth.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_depth.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_depth.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststep_resmlp_normed_depth.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststeppostnormqknorm_transformer_normed_depth.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststepprenormqknorm_transformer_normed_depth.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststeppostnormpreresqknorm_transformer_normed_depth.pt"]
elif args.experiment == "regularupdateablation":
    filenames = ["jan24emawarmup_resmlp_unnormed_depth.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_depth.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_depth.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_depth.pt", "jan27onlyflermfirststepablation_resmlp_normed_depth.pt", "jan27onlyflermfirststepablationpostnormqknorm_transformer_normed_depth.pt", "jan27onlyflermfirststepablationprenormqknorm_transformer_normed_depth.pt", "jan27onlyflermfirststepablationpostnormpreresqknorm_transformer_normed_depth.pt"]

# Call the function for each subplot
for i, ax in enumerate(axes.flat):
    metrics = create_training_plots_on_axis(ax, filename=filenames[i], transferparam="depthmult")

# Only use the legend on one plot
# axes[0, 0].legend()
axes[0, 0].legend(loc='upper left', prop={'size': 8})


# Set the x and y ranges for each plot.
if args.experiment == "regularupdateablation":
    axes[0, 0].set_xlim(5e-6, 5e-3)
    axes[0, 0].set_ylim(0.0002, 0.009)
    axes[0, 1].set_xlim(1e-4, 1e-2)
    axes[0, 1].set_ylim(5.0, 7.0)
    axes[1, 0].set_xlim(5e-6, 5e-3)
    axes[1, 0].set_ylim(0.0002, 0.009)
    axes[1, 1].set_xlim(1e-4, 1e-2)
    axes[1, 1].set_ylim(5.0, 7.0)
    axes[0, 2].set_xlim(1e-4, 0.1)
    axes[0, 2].set_ylim(4.7, 6)
    axes[1, 2].set_xlim(1e-4, 0.1)
    axes[1, 2].set_ylim(4.7, 6)
    axes[0, 3].set_xlim(1e-4, 0.1)
    axes[0, 3].set_ylim(4.68, 6)
    axes[1, 3].set_xlim(1e-4, 0.1)
    axes[1, 3].set_ylim(4.68, 6)
elif args.experiment == "equalmassablation":
    axes[0, 0].set_xlim(5e-6, 5e-3)
    axes[0, 0].set_ylim(0.0002, 0.009)
    axes[0, 1].set_xlim(1e-4, 1e-2)
    axes[0, 1].set_ylim(4.75, 7.0)
    axes[1, 0].set_xlim(5e-3, 5e-0)
    axes[1, 0].set_ylim(0.0002, 0.009)
    axes[1, 1].set_xlim(1e-1, 1e1)
    axes[1, 1].set_ylim(4.75, 7.0)
    axes[0, 2].set_xlim(1e-4, 0.1)
    axes[0, 2].set_ylim(4.7, 6)
    axes[1, 2].set_xlim(3e-1, 300)
    axes[1, 2].set_ylim(4.7, 6)
    axes[0, 3].set_xlim(1e-4, 0.1)
    axes[0, 3].set_ylim(4.68, 6)
    axes[1, 3].set_xlim(3e-1, 300)
    axes[1, 3].set_ylim(4.68, 6)
elif args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    axes[0, 0].set_xlim(5e-6, 5e-3)
    axes[0, 0].set_ylim(0.0002, 0.009)
    axes[0, 1].set_xlim(1e-4, 1e-2)
    axes[0, 1].set_ylim(4.75, 7.0)
    axes[1, 0].set_xlim(5e-3, 5e-0)
    axes[1, 0].set_ylim(0.0002, 0.009)
    axes[1, 1].set_xlim(1e-1, 1e1)
    axes[1, 1].set_ylim(4.75, 7.0)
    axes[0, 2].set_xlim(1e-4, 0.1)
    axes[0, 2].set_ylim(4.7, 6)
    axes[1, 2].set_xlim(1e-2, 1e1)
    axes[1, 2].set_ylim(4.7, 6)
    axes[0, 3].set_xlim(1e-4, 0.1)
    axes[0, 3].set_ylim(4.68, 6)
    axes[1, 3].set_xlim(1e-2, 1e1)
    axes[1, 3].set_ylim(4.68, 6)
elif args.experiment == "onlyflermfirststep":
    axes[0, 0].set_xlim(5e-6, 5e-3)
    axes[0, 0].set_ylim(0.0002, 0.009)
    axes[0, 1].set_xlim(1e-4, 1e-2)
    axes[0, 1].set_ylim(5.0, 7.0)
    axes[1, 0].set_xlim(5e-6, 5e-3)
    axes[1, 0].set_ylim(0.0002, 0.009)
    axes[1, 1].set_xlim(1e-4, 1e-2)
    axes[1, 1].set_ylim(5.0, 7.0)
    axes[0, 2].set_xlim(1e-4, 0.1)
    axes[0, 2].set_ylim(4.7, 6)
    axes[1, 2].set_xlim(1e-4, 0.1)
    axes[1, 2].set_ylim(4.7, 6)
    axes[0, 3].set_xlim(1e-4, 0.1)
    axes[0, 3].set_ylim(4.68, 6)
    axes[1, 3].set_xlim(1e-4, 0.1)
    axes[1, 3].set_ylim(4.68, 6)

# Row labels
# For the first subplot
axes[0, 0].set_ylabel(r"\large Train Loss")
# For the second subplot
axes[1, 0].set_ylabel(r"\large Train Loss")
# Remove the y axis labels for the right plots
axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")
axes[0, 2].set_ylabel("")
axes[1, 2].set_ylabel("")
axes[0, 3].set_ylabel("")
axes[1, 3].set_ylabel("")
# Put x axis labels on the bottom row
axes[1, 0].set_xlabel("\large Learning Rate")
axes[1, 1].set_xlabel("\large Learning Rate")
axes[1, 2].set_xlabel("\large Learning Rate")
axes[1, 3].set_xlabel("\large Learning Rate")

# Put titles "ResMLP" and "Transformer" on the top row
axes[0, 0].set_title(r"\large ResMLP")
axes[0, 1].set_title(r"\large Transformer (PostNorm)")
axes[0, 2].set_title(r"\large Transformer (PreNorm)")
axes[0, 3].set_title(r"\large Transformer (PreNormPostMod)")


# Add a faint background grid to the plots
# for ax in axes.flat:
#     ax.grid(True, linestyle='--', alpha=0.5)

ax2 = axes[0, -1].twinx()
ax2.set_yticks([])
ax2.yaxis.set_tick_params(which='minor', left=False, right=False)
ax2.set_ylabel(r"\large Standard", rotation=270, labelpad=15)
ax2.yaxis.set_label_position("right")
ax2lower = axes[1, -1].twinx()
ax2lower.set_yticks([])
ax2lower.yaxis.set_tick_params(which='minor', left=False, right=False)
if args.experiment == "onlyflermfirststep":
    ax2lower.set_ylabel(r"\large FLeRM", rotation=270, labelpad=15)
elif args.experiment == "regularupdateablation" or args.experiment == "equalmassablation" or args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    ax2lower.set_ylabel(r"FLeRM (ablation, see caption)", rotation=270, labelpad=15)
ax2lower.yaxis.set_label_position("right")

# Set major ticks at every 0.1 on the x axis
resmlp_major_ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
transformer_major_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
axes[0, 0].xaxis.set_major_locator(plt.FixedLocator(resmlp_major_ticks))
axes[0, 1].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[0, 2].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[0, 3].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[1, 0].xaxis.set_major_locator(plt.FixedLocator(resmlp_major_ticks))
axes[1, 1].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[1, 2].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[1, 3].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))

# Set minor ticks at every ie-j on the x axis
resmlp_minor_ticks = [2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1,2e0,3e0,4e0,5e0,6e0,7e0,8e0,9e0,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1]
transformer_minor_ticks = [2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1,2e0,3e0,4e0,5e0,6e0,7e0,8e0,9e0,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,2e2,3e2,4e2,5e2,6e2,7e2,8e2,9e2,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3]
axes[0, 0].xaxis.set_minor_locator(plt.FixedLocator(resmlp_minor_ticks))
axes[0, 1].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[0, 2].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[0, 3].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[1, 0].xaxis.set_minor_locator(plt.FixedLocator(resmlp_minor_ticks))
axes[1, 1].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[1, 2].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[1, 3].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))

plt.tight_layout()
plt.show()

#####################
#WIDTH TRANSFER PLOT#
#####################

# Define a grid
fig, axes = plt.subplots(2, 4, figsize=(12, 5))

if args.experiment == "regularupdateablation":
    filenames = ["jan24emawarmup_resmlp_unnormed_width.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_width.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_width.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_width.pt", "jan24emawarmup_resmlp_normed_width.pt", "jan24emawarmuppostnormqknorm_transformer_normed_width.pt", "jan24emawarmupprenormqknorm_transformer_normed_width.pt", "jan24emawarmuppostnormpreresqknorm_transformer_normed_width.pt"]
elif args.experiment == "equalmassablation":
    filenames = ["jan24emawarmup_resmlp_unnormed_width.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_width.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_width.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_width.pt", "jan28equalmassablationonlyflermfirststep_resmlp_normed_width.pt", "jan28equalmassablationonlyflermfirststeppostnormqknorm_transformer_normed_width.pt", "jan28equalmassablationonlyflermfirststepprenormqknorm_transformer_normed_width.pt", "jan28equalmassablationonlyflermfirststeppostnormpreresqknorm_transformer_normed_width.pt"]
elif args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    filenames = ["jan24emawarmup_resmlp_unnormed_width.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_width.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_width.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_width.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststep_resmlp_normed_width.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststeppostnormqknorm_transformer_normed_width.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststepprenormqknorm_transformer_normed_width.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststeppostnormpreresqknorm_transformer_normed_width.pt"]
elif args.experiment == "onlyflermfirststep":
    filenames = ["jan24emawarmup_resmlp_unnormed_width.pt", "jan24emawarmuppostnormqknorm_transformer_unnormed_width.pt", "jan24emawarmupprenormqknorm_transformer_unnormed_width.pt", "jan24emawarmuppostnormpreresqknorm_transformer_unnormed_width.pt", "jan27onlyflermfirststepablation_resmlp_normed_width.pt", "jan27onlyflermfirststepablationpostnormqknorm_transformer_normed_width.pt", "jan27onlyflermfirststepablationprenormqknorm_transformer_normed_width.pt", "jan27onlyflermfirststepablationpostnormpreresqknorm_transformer_normed_width.pt"]

# Call the function for each subplot
for i, ax in enumerate(axes.flat):
    metrics = create_training_plots_on_axis(ax, filename=filenames[i], transferparam="widthmult")

# Only use the legend on one plot
axes[0, 0].legend()
axes[0, 0].legend(loc='upper left', prop={'size': 8})

# Set the x and y ranges for each plot.
if args.experiment == "regularupdateablation" or args.experiment == "onlyflermfirststep":
    axes[0, 0].set_xlim(1e-6, 0.01)
    axes[0, 0].set_ylim(-0.0005, 0.008)
    axes[0, 1].set_xlim(1e-5, 1e-2)
    axes[0, 1].set_ylim(4.4, 6.5)
    axes[0, 2].set_xlim(1e-5, 0.1)
    axes[0, 2].set_ylim(4.4, 6.5)
    axes[1, 0].set_xlim(1e-6, 0.01)
    axes[1, 0].set_ylim(-0.0005, 0.008)
    axes[1, 1].set_xlim(1e-5, 1e-2)
    axes[1, 1].set_ylim(4.4, 6.5)
    axes[1, 2].set_xlim(1e-5, 0.1)
    axes[1, 2].set_ylim(4.4, 6.5)
    axes[0, 3].set_xlim(1e-5, 0.1)
    axes[0, 3].set_ylim(4.4, 6.5)
    axes[1, 3].set_xlim(1e-5, 0.1)
    axes[1, 3].set_ylim(4.4, 6.5)

elif args.experiment == "equalmassablation" or args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    axes[0, 0].set_xlim(1e-6, 0.01)
    axes[0, 0].set_ylim(-0.0005, 0.008)
    axes[0, 1].set_xlim(1e-5, 1e-2)
    axes[0, 1].set_ylim(4.4, 6.5)
    axes[0, 2].set_xlim(1e-5, 0.1)
    axes[0, 2].set_ylim(4.4, 6.5)
    axes[1, 0].set_xlim(1e-3, 10)
    axes[1, 0].set_ylim(-0.0005, 0.008)
    axes[1, 1].set_xlim(1e-2, 10)
    axes[1, 1].set_ylim(4.4, 6.5)
    axes[1, 2].set_xlim(1e-3, 10)
    axes[1, 2].set_ylim(4.4, 6.5)
    axes[0, 3].set_xlim(1e-5, 0.1)
    axes[0, 3].set_ylim(4.4, 6.5)
    axes[1, 3].set_xlim(1e-3, 10)
    axes[1, 3].set_ylim(4.4, 6.5)

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
axes[0, 0].set_title(r"\large ResMLP")
axes[0, 1].set_title(r"\large Transformer (PostNorm)")
axes[0, 2].set_title(r"\large Transformer (PreNorm)")
axes[0, 3].set_title(r"\large Transformer (PreNormPostMod)")

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
if args.experiment == "onlyflermfirststep":
    ax2lower.set_ylabel(r"\large FLeRM", rotation=270, labelpad=15)
elif args.experiment == "regularupdateablation" or args.experiment == "equalmassablation" or args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    ax2lower.set_ylabel(r"FLeRM (ablation, see caption)", rotation=270, labelpad=15)
ax2lower.yaxis.set_label_position("right")

# Set major ticks at every 0.1 on the x axis
resmlp_major_ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
transformer_major_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
axes[0, 0].xaxis.set_major_locator(plt.FixedLocator(resmlp_major_ticks))
axes[0, 1].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[0, 2].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[0, 3].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[1, 0].xaxis.set_major_locator(plt.FixedLocator(resmlp_major_ticks))
axes[1, 1].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[1, 2].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))
axes[1, 3].xaxis.set_major_locator(plt.FixedLocator(transformer_major_ticks))

# Set minor ticks at every ie-j on the x axis
resmlp_minor_ticks = [2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1,2e0,3e0,4e0,5e0,6e0,7e0,8e0,9e0,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1]
transformer_minor_ticks = [2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,2e-1,3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1,2e0,3e0,4e0,5e0,6e0,7e0,8e0,9e0,2e1,3e1,4e1,5e1,6e1,7e1,8e1,9e1,2e2,3e2,4e2,5e2,6e2,7e2,8e2,9e2,2e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3]
axes[0, 0].xaxis.set_minor_locator(plt.FixedLocator(resmlp_minor_ticks))
axes[0, 1].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[0, 2].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[0, 3].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[1, 0].xaxis.set_minor_locator(plt.FixedLocator(resmlp_minor_ticks))
axes[1, 1].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[1, 2].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))
axes[1, 3].xaxis.set_minor_locator(plt.FixedLocator(transformer_minor_ticks))


plt.tight_layout()
plt.show()

#########################
#INITSCALE TRANSFER PLOT#
#########################

# Define a 2 x 1 grid

# NOTE: I've changed this to a 1 x 2 grid, so none of the comments will make sense anymore.

fig, axes = plt.subplots(1, 2, figsize=(6.1, 3.1))

if args.experiment == "regularupdateablation":
    filenames = ["jan24emawarmuppostnormpreresqknorm_transformer_unnormed_initscale.pt", "jan24emawarmuppostnormpreresqknorm_transformer_normed_initscale.pt"]
elif args.experiment == "equalmassablation":
    filenames = ["jan24emawarmuppostnormpreresqknorm_transformer_unnormed_initscale.pt", "jan28equalmassablationonlyflermfirststeppostnormpreresqknorm_transformer_normed_initscale.pt"]
elif args.experiment == "equalmassbutstillsplittingdepthproperlyablation":
    filenames = ["jan24emawarmuppostnormpreresqknorm_transformer_unnormed_initscale.pt", "jan29equalmassbutstillsplittingdepthproperlyablationonlyflermfirststeppostnormpreresqknorm_transformer_normed_initscale.pt"]
elif args.experiment == "onlyflermfirststep":
    filenames = ["jan24emawarmuppostnormpreresqknorm_transformer_unnormed_initscale.pt", "jan27onlyflermfirststepablationpostnormpreresqknorm_transformer_normed_initscale.pt"]

# Call the function for each subplot
for i, ax in enumerate(axes.flat):
    metrics = create_training_plots_on_axis(ax, filename=filenames[i], transferparam="initscale")

# Only use the legend on one plot
axes[0].legend()
axes[0].legend(loc='upper left', prop={'size': 10})

# Set the x and y ranges for each plot.
if args.experiment == "regularupdateablation" or args.experiment == "onlyflermfirststep":
    axes[0].set_xlim(0.001, 0.1)
    axes[0].set_ylim(4.8, 6)
    axes[1].set_xlim(0.001, 0.1)
    axes[1].set_ylim(4.8, 6)
else:
    axes[0].set_xlim(0.001, 0.1)
    axes[0].set_ylim(4.8, 6)
    axes[1].set_xlim(0.1, 10)
    axes[1].set_ylim(4.8, 6)

# Row labels
# For the first subplot
axes[0].set_ylabel(r"\large{Train Loss}")
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
major_ticks = [0.001, 0.01, 0.1, 1, 10]
axes[0].xaxis.set_major_locator(plt.FixedLocator(major_ticks))
axes[1].xaxis.set_major_locator(plt.FixedLocator(major_ticks))

# Set minor ticks at every ie-j on the x axis
minor_ticks = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 2, 3, 4, 5, 6, 7, 8, 9]
axes[0].xaxis.set_minor_locator(plt.FixedLocator(minor_ticks))
axes[1].xaxis.set_minor_locator(plt.FixedLocator(minor_ticks))

plt.tight_layout()
plt.show()
