import argparse
import numpy as np
from opendataval.dataval import DataShapley
from utils import get_experiment_mediator, SubsetTMCSampler


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, choices=['adult', 'MiniBooNE'])
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
sampler = SubsetTMCSampler(mc_epochs=num_permutations, random_state=seed, cache_name='cached')
data_evaluator = DataShapley(sampler=sampler)
data_evaluator.setup(exper_med.fetcher, exper_med.pred_model, exper_med.metric)

# Generate subset of relevant indices.
print(f'Generating {num_inds}/{num_points} relevant indices...')
relevant_inds = np.random.RandomState(123).choice(num_points, size=num_inds, replace=False)

# Run Monte Carlo estimator.
data_evaluator.train_data_values(relevant_inds=relevant_inds)

# Save results for later aggregation.
results = {
    'estimates': sampler.total_contribution / sampler.total_count,
    'total_contribution': sampler.total_contribution,
    'total_count': sampler.total_count,
    'relevant_inds': relevant_inds
}
np.save(f'data_files/{dataset}/mc-{num_points}-{num_inds}-{seed}.npy', results)
