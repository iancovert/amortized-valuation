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
parser.add_argument('--dataset', type=str, default='cifar10-10')
parser.add_argument('--num_points', type=int, nargs='+', default=[1000, 2500, 5000, 10000, 25000, 49000])
parser.add_argument('--num_contributions', type=int, default=5)
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
ao_error_external = []
ao_expl_var_external = []
ao_corr_external = []
ao_spearman_external = []
ao_sign_external = []

# Prepare ground truth.
filenames = [os.path.join('data_files', dataset, f'mc-50000-500-{seed}.npy')
             for seed in range(10000, 11000)]
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
aggregated_results = aggregate_estimator_results(results)
ground_truth_estimates = torch.tensor(aggregated_results['estimates']).float()
relevant_inds = np.sort(aggregated_results['relevant_inds'])
ground_truth_inds = torch.sort(torch.tensor(aggregated_results['relevant_inds'])).values.long()

# Using training targets, prepare Monte Carlo results with different numbers of samples.
filenames = np.sort(glob(os.path.join('data_files', dataset, 'mc-50000-50000-*.npy')))
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
aggregated_results = aggregate_estimator_results(results[:num_contributions])

# Prepare dataset.
inference_dataset = torch.load(os.path.join('data_files', dataset, 'processed.pt'))['train_dataset']
inference_dataloader = DataLoader(inference_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0)

# For model inference.
model = ResNet18()
trainer = pl.Trainer(
    precision='bf16-mixed',
    accelerator='gpu',
    devices=[0],
    enable_progress_bar=False
)

for num in num_points:
    # Restrict ground truth to relevant points.
    ground_truth_internal = ground_truth_inds[ground_truth_inds < num]
    ground_truth_external = ground_truth_inds[ground_truth_inds >= num]

    # Prepare Monte Carlo estimates.
    estimates = torch.tensor(aggregated_results['estimates']).float()

    # Generate metrics.
    metrics = generate_metrics(estimates[:num], ground_truth_estimates[:num], ground_truth_internal)
    mc_error.append(metrics['error'])
    mc_expl_var.append(metrics['expl_var'])
    mc_corr.append(metrics['corr'])
    mc_spearman.append(metrics['spearman'])
    mc_sign.append(metrics['sign_agreement'])

    # Prepare amortization results with different numbers of samples.
    results = torch.load(os.path.join('model_results', dataset, f'amortized-{num}-{num_contributions}.pt'))
    checkpoint = ValuationModel.load_from_checkpoint(results['best_checkpoint'], model=model)
    estimates = checkpoint.estimates

    # Generate metrics for training points.
    metrics = generate_metrics(estimates, ground_truth_estimates[:num], ground_truth_internal)
    ao_error.append(metrics['error'])
    ao_expl_var.append(metrics['expl_var'])
    ao_corr.append(metrics['corr'])
    ao_spearman.append(metrics['spearman'])
    ao_sign.append(metrics['sign_agreement'])

    # Get predictions for all data points.
    estimates = torch.cat(trainer.predict(checkpoint, inference_dataloader)).squeeze().cpu().float()

    # Generate metrics for external points.
    metrics = generate_metrics(estimates, ground_truth_estimates, ground_truth_external)
    ao_error_external.append(metrics['error'])
    ao_expl_var_external.append(metrics['expl_var'])
    ao_corr_external.append(metrics['corr'])
    ao_spearman_external.append(metrics['spearman'])
    ao_sign_external.append(metrics['sign_agreement'])

# Generate plot showing scaling for different estimators.
fig, axarr = plt.subplots(1, 5, figsize=(20, 4))

# Error.
ax = axarr[0]
ax.plot(num_points, mc_error, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_error, marker='o', label='Amortized')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Error')
ax.set_title('Error Convergence')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Explained variance.
ax = axarr[1]
ax.plot(num_points, mc_expl_var, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_expl_var, marker='o', label='Amortized')
ax.axhline(0, linestyle=':', color='black')
ax.axhline(1, linestyle=':', color='black')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Explained Variance')
ax.set_title('Explained Variance Convergence')
ax.set_ylim([-0.1, 1.1])
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Correlation.
ax = axarr[2]
ax.plot(num_points, mc_corr, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_corr, marker='o', label='Amortized')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Correlation Convergence')
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Spearman.
ax = axarr[3]
ax.plot(num_points, mc_spearman, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_spearman, marker='o', label='Amortized')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Spearman Correlation')
ax.set_title('Rank Correlation Convergence')
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Sign agreement.
ax = axarr[4]
ax.plot(num_points, mc_sign, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_sign, marker='o', label='Amortized')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Sign Agreement')
ax.set_title('Sign Agreement Convergence')
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
filename = f'figures/scaling-{dataset}-{num_contributions}.pdf'
plt.savefig(filename)

print(f'Saved results for {dataset} with {num_contributions} contributions to {filename}')


# Generate plot showing convergence for different estimators.
fig, axarr = plt.subplots(1, 5, figsize=(20, 4))

# Error.
ax = axarr[0]
ax.plot(num_points, mc_error, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_error, marker='o', label='Amortized')
ax.plot(num_points, ao_error_external, marker='o', label='Amortized (External)')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Error')
ax.set_title('Error Convergence')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Explained variance.
ax = axarr[1]
ax.plot(num_points, mc_expl_var, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_expl_var, marker='o', label='Amortized')
ax.plot(num_points, ao_expl_var_external, marker='o', label='Amortized (External)')
ax.axhline(0, linestyle=':', color='black')
ax.axhline(1, linestyle=':', color='black')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Explained Variance')
ax.set_title('Explained Variance Convergence')
ax.set_ylim([-0.1, 1.1])
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Correlation.
ax = axarr[2]
ax.plot(num_points, mc_corr, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_corr, marker='o', label='Amortized')
ax.plot(num_points, ao_corr_external, marker='o', label='Amortized (External)')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Pearson Correlation')
ax.set_title('Correlation Convergence')
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Spearman.
ax = axarr[3]
ax.plot(num_points, mc_spearman, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_spearman, marker='o', label='Amortized')
ax.plot(num_points, ao_spearman_external, marker='o', label='Amortized (External)')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Spearman Correlation')
ax.set_title('Rank Correlation Convergence')
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Sign agreement.
ax = axarr[4]
ax.plot(num_points, mc_sign, marker='o', label='Monte Carlo')
ax.plot(num_points, ao_sign, marker='o', label='Amortized')
ax.plot(num_points, ao_sign_external, marker='o', label='Amortized (External)')
ax.set_xlabel('# Datapoints')
ax.set_ylabel('Sign Agreement')
ax.set_title('Sign Agreement Convergence')
ax.set_xscale('log')
ax.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
filename = f'figures/scaling-{dataset}-external-{num_contributions}.pdf'
plt.savefig(filename)

print(f'Saved results for {dataset} with {num_contributions} points to {filename}')
