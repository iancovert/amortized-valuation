import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

import lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torcheval.metrics.functional import binary_auroc, binary_auprc

from resnet_cifar import ResNet18
from adv import ValuationModel
from utils import aggregate_estimator_results


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="cifar10-10")
parser.add_argument(
    "--mc_contributions", type=int, nargs="+", default=[5, 10, 25, 50, 100]
)
parser.add_argument("--amortized_contributions", type=int, nargs="+", default=[5])
args = parser.parse_args()

# Parse arguments.
dataset = args.dataset
mc_contributions = args.mc_contributions
amortized_contributions = args.amortized_contributions

# Extract noise level.
noise_level = int(dataset.split("-")[-1])

# Load dataset and prepare mislabeled examples.
data_dict = torch.load(os.path.join("data_files", dataset, "processed.pt"))
y_train = data_dict["y_train"]
y_train_true = data_dict["y_train_true"]
mislabeled = y_train != y_train_true

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

# For model inference.
model = ResNet18()
trainer = pl.Trainer(
    precision="bf16-mixed", accelerator="gpu", devices=[0], enable_progress_bar=False
)
inference_dataset = data_dict["train_dataset"]
inference_dataloader = DataLoader(
    inference_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=0
)
inference_dset = TensorDataset(
    torch.stack([inference_dataset[i][0] for i in range(len(inference_dataset))])
)
inference_dloader = DataLoader(
    inference_dset, batch_size=256, shuffle=False, drop_last=False, num_workers=0
)

# Generate Monte Carlo results.
mc_mislabeled_position_dict = {}
filenames = np.sort(glob(os.path.join("data_files", dataset, "mc-50000-50000-*.npy")))
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
mc_estimates = []
for count in mc_contributions:
    # Generate estimates.
    aggregated_results = aggregate_estimator_results(results[:count])
    estimates = torch.tensor(aggregated_results["estimates"]).float()
    mc_estimates.append(estimates)

    # Save mislabeled positions.
    mc_mislabeled_position_dict[count] = mislabeled[torch.argsort(estimates)]

# Generate amortized results.
amortized_mislabeled_position_dict = {}
num_points = 50000
ao_estimates = []
ao_estimates_all = []
for count in amortized_contributions:
    # Load model from checkpoint.
    results = torch.load(
        os.path.join("model_results", dataset, f"amortized-50000-{count}.pt")
    )
    checkpoint = ValuationModel.load_from_checkpoint(
        results["best_checkpoint"], model=model
    )

    # Generate predictions.
    estimates = (
        torch.cat(trainer.predict(checkpoint, inference_dataloader))
        .squeeze()
        .cpu()
        .float()
    )
    ao_estimates.append(estimates)

    # Save mislabeled positions.
    amortized_mislabeled_position_dict[count] = mislabeled[torch.argsort(estimates)]

    # Generate predictions for all classes.
    estimates = (
        torch.cat(trainer.predict(checkpoint, inference_dloader))
        .squeeze()
        .cpu()
        .float()
    )
    ao_estimates_all.append(estimates)

# Load kNN Shapley results.
knnshap_estimates = torch.load(os.path.join("data_files", dataset, "knnshap.pt"))

# Plot jointly.
x_max = 25000
plt.figure()

# Plot perfect identification.
positions = torch.flip(torch.sort(mislabeled).values, (0,))
plt.plot(
    range(1, x_max),
    torch.cumsum(positions.float(), 0)[: x_max - 1],
    label="Ground Truth",
    color="black",
    linestyle=":",
)

# Plot amortized results.
for count, positions in amortized_mislabeled_position_dict.items():
    plt.plot(
        range(1, x_max),
        torch.cumsum(positions.float(), 0)[: x_max - 1],
        label=f"Amortized ({count})",
    )

# Plot Monte Carlo results.
for count, positions in mc_mislabeled_position_dict.items():
    plt.plot(
        range(1, x_max),
        torch.cumsum(positions.float(), 0)[: x_max - 1],
        label=f"Monte Carlo ({count})",
        linestyle="--",
    )

# Plot kNN Shapley results.
positions = mislabeled[torch.argsort(knnshap_estimates)]
plt.plot(
    range(1, x_max),
    torch.cumsum(positions.float(), 0)[: x_max - 1],
    label="kNN Shapley",
)

plt.title(f"CIFAR-10 Mislabeled Example Identification ({noise_level}% Noise)")
plt.xlabel("# Examples Identified")
plt.ylabel("# Mislabeled Examples")
plt.legend(frameon=False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)

plt.tight_layout()
filename = f"figures/mislabeled_{dataset}_50000.pdf"
plt.savefig(filename)

print(f"Saved results for {dataset} to {filename}")


# Plot colored histograms of mislabeled/correct examples.
plt.figure()
mislabeled_scores = ground_truth_estimates[ground_truth_inds][
    mislabeled[ground_truth_inds]
]
correct_scores = ground_truth_estimates[ground_truth_inds][
    ~mislabeled[ground_truth_inds]
]
bins = np.linspace(ground_truth_estimates.min(), ground_truth_estimates.max(), 20)
plt.hist(mislabeled_scores, bins=bins, label="Mislabeled", alpha=0.5, color="tab:red")
plt.hist(correct_scores, bins=bins, label="Correct", alpha=0.5, color="tab:blue")
plt.axvline(x=0, color="black", linestyle=":")
plt.title("Ground Truth Score Distribution")
plt.xlabel("Valuation Score")
plt.ylabel("# Examples")
plt.legend(frameon=False)
plt.tight_layout()
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.tight_layout()
plt.savefig(f"figures/score_distribution_{dataset}_ground_truth.pdf")

for count, estimates in zip(amortized_contributions, ao_estimates):
    plt.figure()
    mislabeled_scores = estimates[mislabeled]
    correct_scores = estimates[~mislabeled]
    bins = np.linspace(estimates.min(), estimates.max(), 100)
    plt.hist(
        mislabeled_scores, bins=bins, label="Mislabeled", alpha=0.5, color="tab:red"
    )
    plt.hist(correct_scores, bins=bins, label="Correct", alpha=0.5, color="tab:blue")
    plt.axvline(x=0, color="black", linestyle=":")
    plt.title(f"Amortized ({count}) Score Distribution")
    plt.xlabel("Valuation Score")
    plt.ylabel("# Examples")
    plt.legend(frameon=False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figures/score_distribution_{dataset}_amortized_{count}.pdf")

for count, estimates in zip(mc_contributions, mc_estimates):
    plt.figure()
    mislabeled_scores = estimates[mislabeled]
    correct_scores = estimates[~mislabeled]
    bins = np.linspace(estimates.min(), estimates.max(), 100)
    plt.hist(
        mislabeled_scores, bins=bins, label="Mislabeled", alpha=0.5, color="tab:red"
    )
    plt.hist(correct_scores, bins=bins, label="Correct", alpha=0.5, color="tab:blue")
    plt.axvline(x=0, color="black", linestyle=":")
    plt.title(f"Monte Carlo ({count}) Score Distribution")
    plt.xlabel("Valuation Score")
    plt.ylabel("# Examples")
    plt.legend(frameon=False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"figures/score_distribution_{dataset}_mc_{count}.pdf")


# Create table of mislabeling statistics.
data = [
    {
        "Overall < 0": (ground_truth_estimates[ground_truth_inds] < 0)
        .float()
        .mean()
        .item(),
        "Mislabeled < 0": (
            ground_truth_estimates[ground_truth_inds][mislabeled[ground_truth_inds]] < 0
        )
        .float()
        .mean()
        .item(),
        "Correct < 0": (
            ground_truth_estimates[ground_truth_inds][~mislabeled[ground_truth_inds]]
            < 0
        )
        .float()
        .mean()
        .item(),
        "Mislabeled AUROC": binary_auroc(
            -ground_truth_estimates[ground_truth_inds],
            mislabeled[ground_truth_inds].int(),
        ).item(),
        "Mislabeled AUPRC": binary_auprc(
            -ground_truth_estimates[ground_truth_inds],
            mislabeled[ground_truth_inds].int(),
        ).item(),
        "Overall max true": 0,
        "Mislabeled max true": 0,
        "Correct max true": 0,
    }
]

# Amortized estimates.
for estimates, estimates_all in zip(ao_estimates, ao_estimates_all):
    data.append(
        {
            "Overall < 0": (estimates < 0).float().mean().item(),
            "Mislabeled < 0": (estimates[mislabeled] < 0).float().mean().item(),
            "Correct < 0": (estimates[~mislabeled] < 0).float().mean().item(),
            "Mislabeled AUROC": binary_auroc(-estimates, mislabeled.int()).item(),
            "Mislabeled AUPRC": binary_auprc(-estimates, mislabeled.int()).item(),
            "Overall max true": (estimates_all.argmax(dim=1) == y_train_true)
            .float()
            .mean()
            .item(),
            "Mislabeled max true": (estimates_all.argmax(dim=1) == y_train_true)[
                mislabeled
            ]
            .float()
            .mean()
            .item(),
            "Correct max true": (estimates_all.argmax(dim=1) == y_train_true)[
                ~mislabeled
            ]
            .float()
            .mean()
            .item(),
        }
    )

# Monte Carlo estimates.
for estimates in mc_estimates:
    data.append(
        {
            "Overall < 0": (estimates < 0).float().mean().item(),
            "Mislabeled < 0": (estimates[mislabeled] < 0).float().mean().item(),
            "Correct < 0": (estimates[~mislabeled] < 0).float().mean().item(),
            "Mislabeled AUROC": binary_auroc(-estimates, mislabeled.int()).item(),
            "Mislabeled AUPRC": binary_auprc(-estimates, mislabeled.int()).item(),
            "Overall max true": 0,
            "Mislabeled max true": 0,
            "Correct max true": 0,
        }
    )

# kNN Shapley.
data.append(
    {
        "Overall < 0": (knnshap_estimates < 0).float().mean().item(),
        "Mislabeled < 0": (knnshap_estimates[mislabeled] < 0).float().mean().item(),
        "Correct < 0": (knnshap_estimates[~mislabeled] < 0).float().mean().item(),
        "Mislabeled AUROC": binary_auroc(-knnshap_estimates, mislabeled.int()).item(),
        "Mislabeled AUPRC": binary_auprc(-knnshap_estimates, mislabeled.int()).item(),
        "Overall max true": 0,
        "Mislabeled max true": 0,
        "Correct max true": 0,
    }
)

# Create dataframe.
index = (
    ["Ground Truth"]
    + [f"Amortized ({count})" for count in amortized_contributions]
    + [f"Monte Carlo ({count})" for count in mc_contributions]
    + ["kNN Shapley"]
)
df = pd.DataFrame(data, index=index)

# Convert to latex.
print(df.T.to_latex(float_format="%.3f"))

# Convert to markdown.
print(df.T.to_markdown(floatfmt=".3f"))
