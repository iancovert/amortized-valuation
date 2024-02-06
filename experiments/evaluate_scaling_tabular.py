import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from adv import ValuationModel
from adv.utils import generate_metrics
from utils import aggregate_estimator_results


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="adult")
parser.add_argument(
    "--num_points", type=int, nargs="+", default=[250, 500, 1000, 2500, 5000, 10000]
)
parser.add_argument("--num_contributions", type=int, default=50)
args = parser.parse_args()

# Parse arguments.
dataset = args.dataset
num_points = args.num_points
num_contributions = args.num_contributions

# For storing results.
mc_error = []
mc_expl_var = []
mc_corr = []
mc_spearman = []
mc_sign = []
ao_error = []
ao_expl_var = []
ao_corr = []
ao_spearman = []
ao_sign = []

for num in num_points:
    # Prepare ground truth.
    filenames = [
        os.path.join("data_files", dataset, f"mc-{num}-{250}-{seed}.npy")
        for seed in range(10000, 11000)
    ]
    results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
    aggregated_results = aggregate_estimator_results(results)
    ground_truth_estimates = torch.tensor(aggregated_results["estimates"]).float()
    relevant_inds = np.sort(aggregated_results["relevant_inds"])
    ground_truth_inds = torch.sort(
        torch.tensor(aggregated_results["relevant_inds"])
    ).values.long()

    # Using training targets, prepare Monte Carlo results with different numbers of samples.
    num_train_files = num_contributions
    filenames = [
        os.path.join("data_files", dataset, f"mc-{num}-{num}-{seed}.npy")
        for seed in range(num_train_files)
    ]
    results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
    aggregated_results = aggregate_estimator_results(results)
    estimates = torch.tensor(aggregated_results["estimates"]).float()

    # Generate metrics.
    metrics = generate_metrics(estimates, ground_truth_estimates, ground_truth_inds)
    mc_error.append(metrics["error"])
    mc_expl_var.append(metrics["expl_var"])
    mc_corr.append(metrics["corr"])
    mc_spearman.append(metrics["spearman"])
    mc_sign.append(metrics["sign_agreement"])

    # Prepare amortization results with different numbers of samples.
    results = torch.load(
        os.path.join(
            "model_results", dataset, f"amortized-{num}-{num_contributions}.pt"
        )
    )
    checkpoint = ValuationModel.load_from_checkpoint(results["best_checkpoint"])
    estimates = checkpoint.estimates

    # Generate metrics.
    metrics = generate_metrics(estimates, ground_truth_estimates, ground_truth_inds)
    ao_error.append(metrics["error"])
    ao_expl_var.append(metrics["expl_var"])
    ao_corr.append(metrics["corr"])
    ao_spearman.append(metrics["spearman"])
    ao_sign.append(metrics["sign_agreement"])

# Generate plot showing scaling for different estimators.
fig, axarr = plt.subplots(1, 5, figsize=(20, 4))

# Error.
ax = axarr[0]
ax.plot(num_points, mc_error, marker="o", label="Monte Carlo")
ax.plot(num_points, ao_error, marker="o", label="Amortized")
ax.set_xlabel("# Datapoints")
ax.set_ylabel("Error")
ax.set_title("Error Convergence")
ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Explained variance.
ax = axarr[1]
ax.plot(num_points, mc_expl_var, marker="o", label="Monte Carlo")
ax.plot(num_points, ao_expl_var, marker="o", label="Amortized")
ax.axhline(0, linestyle=":", color="black")
ax.axhline(1, linestyle=":", color="black")
ax.set_xlabel("# Datapoints")
ax.set_ylabel("Explained Variance")
ax.set_title("Explained Variance Convergence")
ax.set_ylim([-0.1, 1.1])
ax.set_xscale("log")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Correlation.
ax = axarr[2]
ax.plot(num_points, mc_corr, marker="o", label="Monte Carlo")
ax.plot(num_points, ao_corr, marker="o", label="Amortized")
ax.set_xlabel("# Datapoints")
ax.set_ylabel("Pearson Correlation")
ax.set_title("Correlation Convergence")
ax.set_xscale("log")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Spearman.
ax = axarr[3]
ax.plot(num_points, mc_spearman, marker="o", label="Monte Carlo")
ax.plot(num_points, ao_spearman, marker="o", label="Amortized")
ax.set_xlabel("# Datapoints")
ax.set_ylabel("Spearman Correlation")
ax.set_title("Rank Correlation Convergence")
ax.set_xscale("log")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Sign agreement.
ax = axarr[4]
ax.plot(num_points, mc_sign, marker="o", label="Monte Carlo")
ax.plot(num_points, ao_sign, marker="o", label="Amortized")
ax.set_xlabel("# Datapoints")
ax.set_ylabel("Sign Agreement")
ax.set_title("Sign Agreement Convergence")
ax.set_xscale("log")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()
filename = f"figures/scaling-{dataset}-{num_contributions}.pdf"
plt.savefig(filename)

print(
    f"Saved results for {dataset} with {num_contributions} contributions to {filename}"
)
