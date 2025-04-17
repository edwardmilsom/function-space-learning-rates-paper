import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
kronecker_samples = torch.load("comparevarianceskronecker_postnormpreres_transformersamplesforplottingvariance_lr_0.008577119235556976_seed_1.ptnormsforvariances")
noassumption_samples = torch.load("comparevariancesfull_cov_postnormpreres_transformersamplesforplottingvariance_lr_0.008577119235556976_seed_1.ptnormsforvariances")
iid_samples = torch.load("comparevariancesiid_postnormpreres_transformersamplesforplottingvariance_lr_0.008577119235556976_seed_1.ptnormsforvariances")

# Each _samples file is a dicitonary of lists, i.e. the keys are layers of the NN, and the values are lists of samples

layernames = list(kronecker_samples.keys())

# Plot KDE of each method for first layer
for layer in layernames:
    # sns.kdeplot(kronecker_samples[layer], label="Kronecker", color='r', linewidth=3) # Sometimes Kronecker overlaps with either IID or No Assumption, so make it thicker and on the bottom
    # sns.kdeplot(iid_samples[layer], label="IID", color='g')
    # sns.kdeplot(noassumption_samples[layer], label="No Assumption", color='b')

    # # Also have a vertical dashed line for mean of each method
    # plt.axvline(torch.mean(torch.tensor(kronecker_samples[layer]).pow(2)).pow(0.5), color='r', linestyle='--', linewidth=3)
    # plt.axvline(torch.mean(torch.tensor(iid_samples[layer]).pow(2)).pow(0.5), color='g', linestyle='--')
    # plt.axvline(torch.mean(torch.tensor(noassumption_samples[layer]).pow(2)).pow(0.5), color='b', linestyle='--')

    # plt.title(f"Layer {layer}")
    # plt.xlabel("FSLR Estimate")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.show()

    ### TODO: FIX SQRT SQUARE EXPECTATION ISSUE

    # # Also plot the mean and standard deviation on a separate plot, one column for each method, showing the mean as a point and the standard deviation as a line
    # plt.errorbar(0, torch.mean(torch.tensor(iid_samples[layer])), yerr=torch.std(torch.tensor(iid_samples[layer])), fmt='o', color='g')
    # plt.errorbar(1, torch.mean(torch.tensor(kronecker_samples[layer])), yerr=torch.std(torch.tensor(kronecker_samples[layer])), fmt='o', color='r')
    # plt.errorbar(2, torch.mean(torch.tensor(noassumption_samples[layer])), yerr=torch.std(torch.tensor(noassumption_samples[layer])), fmt='o', color='b')
    # plt.xticks([0, 1, 2], ['IID', 'KFAC', 'No Assumption'])
    # plt.ylabel("FSLR Estimate")
    # plt.xlabel("Approximation Assumption")
    # plt.title(f"Layer {layer}")
    # plt.show()

    # # Do a figure with 2 subplots, one bar chart for the bias (mean minus the mean for no assumption), and one for the variance.
    fig, axs = plt.subplots(1, 2)

    # Set figure size
    fig.set_size_inches(6, 3)

    axs[0].bar(0, torch.abs(torch.mean(torch.tensor(noassumption_samples[layer]).pow(2)).pow(0.5) - torch.mean(torch.tensor(noassumption_samples[layer]).pow(2)).pow(0.5)), color='b')
    axs[0].bar(1, torch.abs(torch.mean(torch.tensor(iid_samples[layer]).pow(2)).pow(0.5) - torch.mean(torch.tensor(noassumption_samples[layer]).pow(2)).pow(0.5)), color='g')
    axs[0].bar(2, torch.abs(torch.mean(torch.tensor(kronecker_samples[layer]).pow(2)).pow(0.5) - torch.mean(torch.tensor(noassumption_samples[layer]).pow(2)).pow(0.5)), color='r')
    axs[0].set_xticks([0, 1, 2])
    axs[0].set_xticklabels(['No Assumption', 'IID', 'KFAC'])
    axs[0].set_ylabel("|Bias|")
    # Also add standard error of mean as error bars
    # axs[0].errorbar(0, torch.abs(torch.mean(torch.tensor(noassumption_samples[layer])) - torch.mean(torch.tensor(noassumption_samples[layer]))), yerr=torch.std(torch.tensor(noassumption_samples[layer])) / (len(noassumption_samples[layer])**0.5), fmt='o', color='b')
    # axs[0].errorbar(1, torch.abs(torch.mean(torch.tensor(iid_samples[layer])) - torch.mean(torch.tensor(noassumption_samples[layer]))), yerr=torch.std(torch.tensor(iid_samples[layer])) / (len(iid_samples[layer])**0.5), fmt='o', color='g')
    # axs[0].errorbar(2, torch.abs(torch.mean(torch.tensor(kronecker_samples[layer])) - torch.mean(torch.tensor(noassumption_samples[layer]))), yerr=torch.std(torch.tensor(kronecker_samples[layer])) / (len(kronecker_samples[layer])**0.5), fmt='o', color='r')
    # axs[0].set_title(f"Layer {layer}")

    axs[1].bar(0, torch.var(torch.tensor(noassumption_samples[layer])), color='b')
    axs[1].bar(1, torch.var(torch.tensor(iid_samples[layer])), color='g')
    axs[1].bar(2, torch.var(torch.tensor(kronecker_samples[layer])), color='r')
    axs[1].set_xticks([0, 1, 2])
    axs[1].set_xticklabels(['No Assumption', 'IID', 'KFAC'])
    axs[1].set_ylabel("Variance")
    # axs[1].set_title(f"Layer {layer}")

    # Put a title for the whole figure
    fig.suptitle(f"Layer {layer}")

    plt.tight_layout()
    plt.show()
