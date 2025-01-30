import matplotlib.pyplot as plt
import torch
import argparse

# Define --normalise flag
parser = argparse.ArgumentParser()
parser.add_argument('--normalise', action='store_true', help='Normalise parameter values at each iteration')
args = parser.parse_args()

def plot_parameter_values(param_dict, highlight_groups=None, normalise=False, ax=None, lr=1):
    """
    Plot parameter values across iterations with parameter grouping.
    
    Args:
        param_dict (dict): Dictionary with parameter names as keys 
                            and lists of tensor values as values
        highlight_groups (dict, optional): Dictionary mapping display names to lists
                                         of parameter names to highlight together
        normalise (bool, optional): Whether to normalise values at each iteration
        ax (matplotlib.axes.Axes, optional): Axis to plot on. 
                                             Creates new figure if None.
    """
    # If no axis provided, create a new figure
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Default to no highlights if not specified
    if highlight_groups is None:
        highlight_groups = {}
        
    # Create set of all highlighted parameters for quick lookup
    highlighted_params = {
        param 
        for param_list in highlight_groups.values() 
        for param in param_list
    }
    
    # Determine number of iterations
    num_iters = len(next(iter(param_dict.values())))
    
    # normalise values if requested
    if normalise:
        # normalise values at each iteration
        plot_dict = {}
        for iteration in range(num_iters):
            # Collect values for this iteration
            iter_values = {
                param: param_values[iteration].item() 
                for param, param_values in param_dict.items()
            }
            
            # Calculate total sum for this iteration
            total_sum = sum(iter_values.values())
            
            # normalise values
            normalised_values = {
                param: val/total_sum for param, val in iter_values.items() # Note lr doesn't matter here because it would disappear when we normalise anyway
            }
            
            # Store normalised values
            for param, norm_val in normalised_values.items():
                if param not in plot_dict:
                    plot_dict[param] = []
                plot_dict[param].append(norm_val)
    else:
        # Use original values if not normalizing
        plot_dict = {
            param: [val.item()*lr for val in values] 
            for param, values in param_dict.items()
        }
    
    # Track whether "Other layers" has been plotted
    other_layers_plotted = False
    
    # First plot non-highlighted parameters
    for param_name, values in plot_dict.items():
        if param_name not in highlighted_params:
            # x_values = list(range(len(values)))
            x_values = [i*100 for i in range(len(values))] # We only recorded every 100 iterations
            if not other_layers_plotted:
                ax.plot(x_values, values, label="Other layers", color='lightgrey', alpha=0.5, linewidth=2)
                other_layers_plotted = True
            else:
                ax.plot(x_values, values, color='lightgrey', alpha=0.5, linewidth=2)
    
    # Then plot each highlight group
    for group_name, param_list in highlight_groups.items():
        # Use same color for all parameters in group
        color = next(ax._get_lines.prop_cycler)['color']
        
        # Plot each parameter in the group
        for param_name in param_list:
            if param_name in plot_dict:  # Check if parameter exists
                # x_values = list(range(len(plot_dict[param_name])))
                x_values = [i*100 for i in range(len(plot_dict[param_name]))] # We only recorded every 100 iterations
                if param_name == param_list[0]:  # Only add to legend once per group
                    ax.plot(x_values, plot_dict[param_name], label=group_name, 
                           color=color, linewidth=2)
                else:
                    ax.plot(x_values, plot_dict[param_name], color=color, linewidth=2)
    
    # Add labels and legend
    ax.set_xlabel(r'\large Iteration')
    y_label = r'\large Relative Function-Space LR' if normalise else r'\large Function-Space LR'
    ax.set_ylabel(y_label)
    #Make y axis log scale
    ax.set_yscale('log')
    ax.legend()
    # Put legend in top left
    ax.legend(loc='upper left')

    # Add a faint grid
    ax.grid(True, which='both', alpha=0.3)
    
    return ax

# Example usage
# if __name__ == "__main__":
#     # Example dictionary matching the format in the question
#     example_dict = {
#         "layer.0.weight": [torch.tensor([0.223]), torch.tensor([0.4567])],
#         "layer.1.weight": [torch.tensor([1.945]), torch.tensor([2.365])],
#         "layer.2.weight": [torch.tensor([0.567]), torch.tensor([0.789])],
#         "layer.3.weight": [torch.tensor([1.234]), torch.tensor([1.456])]
#     }
    
#     # Example highlight groups
#     highlight_groups = {
#         "Input layers": ["layer.0.weight", "layer.1.weight"],
#         "Output layers": ["layer.2.weight", "layer.3.weight"]
#     }
    
#     plot_parameter_values(example_dict, highlight_groups=highlight_groups, normalise=False)
#     plt.show()

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'''
\usepackage{helvet}  % Load Helvetica font
\renewcommand{\familydefault}{\sfdefault}  % Use sans-serif as default
'''

fig, ax = plt.subplots(2,1, figsize=(6, 6))

resmlp_dict = torch.load("resmlp_fslrs.ptnormsforplotting", map_location=torch.device('cpu'))
postnormtransformer_dict = torch.load("postnormtransformer_fslrs.ptnormsforplotting", map_location=torch.device('cpu'))
prenormtransformer_dict = torch.load("prenormtransformer_fslrs.ptnormsforplotting", map_location=torch.device('cpu'))
postnormprerestransformer_dict = torch.load("postnormprerestransformer_fslrs.ptnormsforplotting", map_location=torch.device('cpu'))

print(resmlp_dict.keys())
print(postnormtransformer_dict.keys())

resmlp_groups = {
    "Biases": ["layers.0.bias", "layers.1.0.0.bias", "layers.1.1.0.bias", "layers.1.2.0.bias", "layers.1.3.0.bias", "layers.2.bias"],
    "Input": ["layers.0.weight"],
    "Output": ["layers.2.weight"],
    "Hidden": ["layers.1.0.0.weight", "layers.1.1.0.weight", "layers.1.2.0.weight", "layers.1.3.0.weight"]
}

# transformer_groups = {
#     "Biases": [
#         "layers.0.self_attn.q_proj.bias", "layers.0.self_attn.k_proj.bias", 
#         "layers.0.self_attn.v_proj.bias", "layers.0.self_attn.out_proj.bias",
#         "layers.0.ff.net.0.bias", "layers.0.ff.net.2.bias",
#         "layers.1.self_attn.q_proj.bias", "layers.1.self_attn.k_proj.bias",
#         "layers.1.self_attn.v_proj.bias", "layers.1.self_attn.out_proj.bias",
#         "layers.1.ff.net.0.bias", "layers.1.ff.net.2.bias",
#         "output_proj.bias"
#     ],
#     "Q Weights": ["layers.0.self_attn.q_proj.weight", "layers.1.self_attn.q_proj.weight"],
#     "K Weights": ["layers.0.self_attn.k_proj.weight", "layers.1.self_attn.k_proj.weight"],
#     "V Weights": ["layers.0.self_attn.v_proj.weight", "layers.1.self_attn.v_proj.weight"],
#     "O Weights": ["layers.0.self_attn.out_proj.weight", "layers.1.self_attn.out_proj.weight"],
#     "FF Weights 1": ["layers.0.ff.net.0.weight", "layers.1.ff.net.0.weight"],
#     "FF Weights 2": ["layers.0.ff.net.2.weight", "layers.1.ff.net.2.weight"],
#     "Readout": ["output_proj.weight"],
#     "Embedding": ["input_emb.weight"]
# }

transformer_groups = {
    "Biases": [
        "layers.0.self_attn.q_proj.bias", "layers.0.self_attn.k_proj.bias", 
        "layers.0.self_attn.v_proj.bias", "layers.0.self_attn.out_proj.bias",
        "layers.0.ff.net.0.bias", "layers.0.ff.net.2.bias",
        "layers.1.self_attn.q_proj.bias", "layers.1.self_attn.k_proj.bias",
        "layers.1.self_attn.v_proj.bias", "layers.1.self_attn.out_proj.bias",
        "layers.1.ff.net.0.bias", "layers.1.ff.net.2.bias",
        "output_proj.bias"
    ],
    "Embedding": ["input_emb.weight"],
    "Readout": ["output_proj.weight"],
    "QK Weights": ["layers.0.self_attn.q_proj.weight", "layers.1.self_attn.q_proj.weight","layers.0.self_attn.k_proj.weight", "layers.1.self_attn.k_proj.weight"],
    "VO Weights": ["layers.0.self_attn.v_proj.weight", "layers.1.self_attn.v_proj.weight","layers.0.self_attn.out_proj.weight", "layers.1.self_attn.out_proj.weight"],
    "FF Weights 1": ["layers.0.ff.net.0.weight", "layers.1.ff.net.0.weight"],
    "FF Weights 2": ["layers.0.ff.net.2.weight", "layers.1.ff.net.2.weight"]
}

# plot_parameter_values(resmlp_dict, highlight_params=["layers.0.weight", "layers.2.weight"], ax=ax[0], lr=0.0004641609040922991, normalise=args.normalise)
# plot_parameter_values(postnormtransformer_dict, highlight_params=["input_emb.weight", "output_proj.weight"], ax=ax[1], lr=0.004641691597003633, normalise=args.normalise)
# plot_parameter_values(prenormtransformer_dict, highlight_params=["input_emb.weight", "output_proj.weight"], ax=ax[2], lr=0.008577119235556976, normalise=args.normalise)
# plot_parameter_values(postnormprerestransformer_dict, highlight_params=["input_emb.weight", "output_proj.weight"], ax=ax[3], lr=0.008577119235556976, normalise=args.normalise)
plot_parameter_values(resmlp_dict, highlight_groups=resmlp_groups, normalise=args.normalise, ax=ax[0], lr = 0.0004641609040922991)
plot_parameter_values(postnormtransformer_dict, highlight_groups=transformer_groups, normalise=args.normalise, ax=ax[1], lr = 0.004641691597003633)
# plot_parameter_values(prenormtransformer_dict, highlight_groups=transformer_groups, normalise=args.normalise, ax=ax[2], lr = 0.008577119235556976)
# plot_parameter_values(postnormprerestransformer_dict, highlight_groups=transformer_groups, normalise=args.normalise, ax=ax[3], lr = 0.008577119235556976)

ax2 = []
modelnames = ["ResMLP", "Transformer (PostNorm)", "Transformer (PreNorm)", "Transformer (PreNormPostMod)"]
for i in range(2):
    ax2.append(ax[i].twinx())
    ax2[i].set_yticks([])
    ax2[i].yaxis.set_tick_params(which='minor', left=False, right=False)
    # ax2[i].set_ylabel(r"\large Standard", rotation=270, labelpad=15)
    ax2[i].set_ylabel(r"\large " + modelnames[i], rotation=270, labelpad=15)
    ax2[i].yaxis.set_label_position("right")

plt.tight_layout()
plt.show()