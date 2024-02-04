import argparse
import numpy as np
from opendataval.dataval import DataShapley
from utils import get_experiment_mediator, SubsetDistributionalSampler


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, choices=['cifar10-10', 'cifar10-25', 'cifar10-50'])
parser.add_argument('-num_points', type=int)
parser.add_argument('-num_inds', type=int)
parser.add_argument('-num_permutations', type=int, default=10)
parser.add_argument('-seed', type=int)

# Parse arguments.
args = parser.parse_args()
dataset = args.dataset
num_points = args.num_points
num_inds = args.num_inds
num_permutations = args.num_permutations
seed = args.seed

# Get experiment mediator and set up data evaluator.
exper_med = get_experiment_mediator(dataset, num_points)
sampler = SubsetDistributionalSampler(
    mc_epochs=num_permutations, min_cardinality=100, max_cardinality=1000,
    random_state=seed, cache_name='cached')
data_evaluator = DataShapley(sampler=sampler)
data_evaluator.setup(exper_med.fetcher, exper_med.pred_model, exper_med.metric)

# Generate subset of relevant indices.
print(f'Generating {num_inds}/{num_points} relevant indices...')
num_points = len(exper_med.fetcher.x_train)
if num_inds == num_points:
    relevant_inds = np.arange(num_points)
else:
    # Select inds at the beginning and end.
    relevant_inds = np.concatenate([np.arange(num_inds // 2), np.arange(num_points - num_inds // 2, num_points)])

# Set random seed and run Monte Carlo estimator.
data_evaluator.train_data_values(relevant_inds=relevant_inds)

# Save results for later aggregation.
results = {
    'estimates': sampler.total_contribution / sampler.total_count,
    'total_contribution': sampler.total_contribution,
    'total_count': sampler.total_count,
    'relevant_inds': relevant_inds
}
np.save(f'data_files/{dataset}/mc-{num_points}-{num_inds}-{seed}.npy', results)
