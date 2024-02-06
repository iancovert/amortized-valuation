import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

from adv.utils import generate_metrics
from utils import aggregate_estimator_results


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="adult")
parser.add_argument("--num_points", type=int, default=1000)
args = parser.parse_args()

# Parse arguments.
dataset = args.dataset
num_points = args.num_points

# Load and partition Monte Carlo results.
all_filenames = glob(os.path.join("data_files", dataset, f"mc-{num_points}-*.npy"))
train_filenames = glob(
    os.path.join("data_files", dataset, f"mc-{num_points}-{num_points}-*.npy")
)
gt_filenames = [
    filename for filename in all_filenames if filename not in train_filenames
]
gt_first = gt_filenames[: len(gt_filenames) // 2]
gt_second = gt_filenames[len(gt_filenames) // 2 :]
assert len(train_filenames) > 10
assert len(gt_filenames) > 10
gt_first_results = [
    np.load(filename, allow_pickle=True).item() for filename in gt_first
]
gt_second_results = [
    np.load(filename, allow_pickle=True).item() for filename in gt_second
]
train_results = [
    np.load(filename, allow_pickle=True).item() for filename in train_filenames
]

# Determine number of samples to display.
samples_per_gt = int(max([result["total_count"].max() for result in gt_first_results]))
samples_per_train = int(max([result["total_count"].max() for result in train_results]))
assert 10 % samples_per_gt == 0
assert 10 % samples_per_train == 0
total_gt_samples = len(gt_second) * samples_per_gt
total_train_samples = len(train_filenames) * samples_per_train
max_samples = min(total_gt_samples, total_train_samples)
sample_counts = np.arange(10, max_samples + 10, 10)

# Prepare ground truth with first half.
aggregated_results = aggregate_estimator_results(gt_first_results)
ground_truth_estimates = torch.tensor(aggregated_results["estimates"]).float()
relevant_inds = np.sort(aggregated_results["relevant_inds"])
ground_truth_inds = torch.sort(
    torch.tensor(aggregated_results["relevant_inds"])
).values.long()

# Using ground truth runs, prepare estimators with different numbers of samples.
gt_error = []
gt_expl_var = []
gt_corr = []
for count in sample_counts:
    # Aggregate corresponding results.
    aggregated_results = aggregate_estimator_results(
        gt_second_results[: count // samples_per_gt]
    )
    estimates = torch.tensor(aggregated_results["estimates"]).float()

    # Generate metrics.
    metrics = generate_metrics(estimates, ground_truth_estimates, ground_truth_inds)
    gt_error.append(metrics["error"])
    gt_expl_var.append(metrics["expl_var"])
    gt_corr.append(metrics["corr"])

# Using training data runs, prepare estimators with different numbers of samples.
train_error = []
train_expl_var = []
train_corr = []
for count in sample_counts:
    # Aggregate corresponding results.
    aggregated_results = aggregate_estimator_results(
        train_results[: count // samples_per_train]
    )
    estimates = torch.tensor(aggregated_results["estimates"]).float()

    # Generate metrics.
    metrics = generate_metrics(estimates, ground_truth_estimates, ground_truth_inds)
    train_error.append(metrics["error"])
    train_expl_var.append(metrics["expl_var"])
    train_corr.append(metrics["corr"])

# Generate plot showing convergence for different Monte Carlo runs.
fig, axarr = plt.subplots(1, 3, figsize=(12, 4))

# Error.
ax = axarr[0]
ax.plot(sample_counts, gt_error, label="Ground truth runs")
ax.plot(sample_counts, train_error, label="Training data runs")
ax.set_xlabel("# Samples")
ax.set_ylabel("Error")
ax.set_title("Error Convergence")
ax.legend()

# Explained variance.
ax = axarr[1]
ax.plot(sample_counts, gt_expl_var, label="Ground truth runs")
ax.plot(sample_counts, train_expl_var, label="Training data runs")
ax.set_xlabel("# Samples")
ax.set_ylabel("Explained Variance")
ax.set_title("Explained Variance Convergence")
ax.legend()

# Correlation.
ax = axarr[2]
ax.plot(sample_counts, gt_corr, label="Ground truth runs")
ax.plot(sample_counts, train_corr, label="Training data runs")
ax.set_xlabel("# Samples")
ax.set_ylabel("Pearson Correlation")
ax.set_title("Correlation Convergence")
ax.legend()

plt.tight_layout()
filename = f"figures/sanity-{dataset}-{num_points}.pdf"
plt.savefig(filename)

print(f"Saved results for {dataset} with {num_points} points to {filename}")
