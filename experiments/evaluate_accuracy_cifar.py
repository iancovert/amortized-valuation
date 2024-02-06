import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import lightning as pl
from torch.utils.data import DataLoader

from resnet_cifar import ResNet18
from adv import ValuationModel
from adv.utils import generate_metrics
from utils import aggregate_estimator_results


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10-10")
parser.add_argument("--num_points", type=int, default=49000)
parser.add_argument(
    "--num_contributions", type=int, nargs="+", default=[5, 10, 15, 20, 25]
)
args = parser.parse_args()

# Parse arguments.
dataset = args.dataset
num_points = args.num_points
num_contributions = args.num_contributions

# Prepare ground truth.
filenames = [
    os.path.join("data_files", dataset, f"mc-50000-500-{seed}.npy")
    for seed in range(10000, 11000)
]
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
aggregated_results = aggregate_estimator_results(results)
ground_truth_estimates = torch.tensor(aggregated_results["estimates"]).float()
relevant_inds = np.sort(aggregated_results["relevant_inds"])
ground_truth_inds = torch.sort(
    torch.tensor(aggregated_results["relevant_inds"])
).values.long()

# Restrict ground truth to relevant points.
ground_truth_internal = ground_truth_inds[ground_truth_inds < num_points]
ground_truth_external = ground_truth_inds[ground_truth_inds >= num_points]

# Prepare dataset.
inference_dataset = torch.load(os.path.join("data_files", dataset, "processed.pt"))[
    "train_dataset"
]
inference_dataloader = DataLoader(
    inference_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0
)

# For model inference.
trainer = pl.Trainer(
    precision="bf16-mixed", accelerator="gpu", devices=[0], enable_progress_bar=False
)

# Using training targets, prepare Monte Carlo results with different numbers of samples.
mc_error = []
mc_expl_var = []
mc_corr = []
mc_spearman = []
mc_sign = []
filenames = np.sort(glob(os.path.join("data_files", dataset, "mc-50000-50000-*.npy")))
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
for count in num_contributions:
    # Get corresponding estimates.
    aggregated_results = aggregate_estimator_results(results[:count])
    estimates = torch.tensor(aggregated_results["estimates"]).float()

    # Generate metrics.
    metrics = generate_metrics(
        estimates[:num_points],
        ground_truth_estimates[:num_points],
        ground_truth_internal,
    )
    mc_error.append(metrics["error"])
    mc_expl_var.append(metrics["expl_var"])
    mc_corr.append(metrics["corr"])
    mc_spearman.append(metrics["spearman"])
    mc_sign.append(metrics["sign_agreement"])

# Prepare amortization results with different numbers of samples.
model = ResNet18()
ao_error = []
ao_expl_var = []
ao_corr = []
ao_spearman = []
ao_sign = []
ao_error_external = []
ao_expl_var_external = []
ao_corr_external = []
ao_spearman_external = []
ao_sign_external = []
for count in num_contributions:
    # Load estimates from best model.
    results = torch.load(
        os.path.join("model_results", dataset, f"amortized-{num_points}-{count}.pt")
    )
    checkpoint = ValuationModel.load_from_checkpoint(
        results["best_checkpoint"], model=model
    )
    estimates = checkpoint.estimates

    # Generate metrics for training points.
    metrics = generate_metrics(
        estimates, ground_truth_estimates[:num_points], ground_truth_internal
    )
    ao_error.append(metrics["error"])
    ao_expl_var.append(metrics["expl_var"])
    ao_corr.append(metrics["corr"])
    ao_spearman.append(metrics["spearman"])
    ao_sign.append(metrics["sign_agreement"])

    if num_points < 50000:
        # Get predictions.
        estimates = (
            torch.cat(trainer.predict(checkpoint, inference_dataloader))
            .squeeze()
            .cpu()
            .float()
        )

        # Generate metrics for external points.
        metrics = generate_metrics(
            estimates, ground_truth_estimates, ground_truth_external
        )
        ao_error_external.append(metrics["error"])
        ao_expl_var_external.append(metrics["expl_var"])
        ao_corr_external.append(metrics["corr"])
        ao_spearman_external.append(metrics["spearman"])
        ao_sign_external.append(metrics["sign_agreement"])


# ------------------------------
# Plot results (internal data)
# ------------------------------

# Generate plot showing convergence for different estimators.
fig, axarr = plt.subplots(1, 5, figsize=(20, 4))

# Error.
ax = axarr[0]
ax.plot(num_contributions, mc_error, marker="o", label="Monte Carlo")
ax.plot(num_contributions, ao_error, marker="o", label="Amortized")
ax.set_xlabel("# Contributions / Point")
ax.set_ylabel("Error")
ax.set_title("Error Convergence")
ax.set_yscale("log")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Explained variance.
ax = axarr[1]
ax.plot(num_contributions, mc_expl_var, marker="o", label="Monte Carlo")
ax.plot(num_contributions, ao_expl_var, marker="o", label="Amortized")
ax.axhline(0, linestyle=":", color="black")
ax.axhline(1, linestyle=":", color="black")
ax.set_xlabel("# Contributions / Point")
ax.set_ylabel("Explained Variance")
ax.set_title("Explained Variance Convergence")
ax.set_ylim([-0.1, 1.1])
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Correlation.
ax = axarr[2]
ax.plot(num_contributions, mc_corr, marker="o", label="Monte Carlo")
ax.plot(num_contributions, ao_corr, marker="o", label="Amortized")
ax.set_xlabel("# Contributions / Point")
ax.set_ylabel("Pearson Correlation")
ax.set_title("Correlation Convergence")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Spearman.
ax = axarr[3]
ax.plot(num_contributions, mc_spearman, marker="o", label="Monte Carlo")
ax.plot(num_contributions, ao_spearman, marker="o", label="Amortized")
ax.set_xlabel("# Contributions / Point")
ax.set_ylabel("Spearman Correlation")
ax.set_title("Rank Correlation Convergence")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# Sign agreement.
ax = axarr[4]
ax.plot(num_contributions, mc_sign, marker="o", label="Monte Carlo")
ax.plot(num_contributions, ao_sign, marker="o", label="Amortized")
ax.set_xlabel("# Contributions / Point")
ax.set_ylabel("Sign Agreement")
ax.set_title("Sign Agreement Convergence")
ax.legend(frameon=False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()
filename = f"figures/accuracy-{dataset}-{num_points}.pdf"
plt.savefig(filename)

print(f"Saved results for {dataset} with {num_points} points to {filename}")


# ------------------------------
# Plot results (external data)
# ------------------------------

if num_points < 50000:
    # Generate plot showing convergence for different estimators.
    fig, axarr = plt.subplots(1, 5, figsize=(20, 4))

    # Error.
    ax = axarr[0]
    ax.plot(num_contributions, mc_error, marker="o", label="Monte Carlo")
    ax.plot(num_contributions, ao_error, marker="o", label="Amortized")
    ax.plot(
        num_contributions, ao_error_external, marker="o", label="Amortized (External)"
    )
    ax.set_xlabel("# Contributions / Point")
    ax.set_ylabel("Error")
    ax.set_title("Error Convergence")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Explained variance.
    ax = axarr[1]
    ax.plot(num_contributions, mc_expl_var, marker="o", label="Monte Carlo")
    ax.plot(num_contributions, ao_expl_var, marker="o", label="Amortized")
    ax.plot(
        num_contributions,
        ao_expl_var_external,
        marker="o",
        label="Amortized (External)",
    )
    ax.axhline(0, linestyle=":", color="black")
    ax.axhline(1, linestyle=":", color="black")
    ax.set_xlabel("# Contributions / Point")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Explained Variance Convergence")
    ax.set_ylim([-0.1, 1.1])
    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Correlation.
    ax = axarr[2]
    ax.plot(num_contributions, mc_corr, marker="o", label="Monte Carlo")
    ax.plot(num_contributions, ao_corr, marker="o", label="Amortized")
    ax.plot(
        num_contributions, ao_corr_external, marker="o", label="Amortized (External)"
    )
    ax.set_xlabel("# Contributions / Point")
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Correlation Convergence")
    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Spearman.
    ax = axarr[3]
    ax.plot(num_contributions, mc_spearman, marker="o", label="Monte Carlo")
    ax.plot(num_contributions, ao_spearman, marker="o", label="Amortized")
    ax.plot(
        num_contributions,
        ao_spearman_external,
        marker="o",
        label="Amortized (External)",
    )
    ax.set_xlabel("# Contributions / Point")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title("Rank Correlation Convergence")
    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Sign agreement.
    ax = axarr[4]
    ax.plot(num_contributions, mc_sign, marker="o", label="Monte Carlo")
    ax.plot(num_contributions, ao_sign, marker="o", label="Amortized")
    ax.plot(
        num_contributions, ao_sign_external, marker="o", label="Amortized (External)"
    )
    ax.set_xlabel("# Contributions / Point")
    ax.set_ylabel("Sign Agreement")
    ax.set_title("Sign Agreement Convergence")
    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    filename = f"figures/accuracy-{dataset}-external-{num_points}.pdf"
    plt.savefig(filename)

    print(f"Saved results for {dataset} with {num_points} points to {filename}")
