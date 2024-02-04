# Amortized data valuation

This repository implements an amortized approach to data valuation from [this](https://arxiv.org/abs/2401.15866) paper. The idea is to learn a neural network that predicts data valuation scores from the original data point (e.g., images), and to train the model efficiently using noisy Monte Carlo estimates as prediction targets.

## Usage

To run this approach, you'll need to do the following:

1. Generate initial data value estimates using a Monte Carlo sampling approach (one of [these](https://github.com/iancovert/amortized-valuation/blob/main/experiments/monte_carlo.py) [two](https://github.com/iancovert/amortized-valuation/blob/main/experiments/monte_carlo_distributional.py) scripts). You should generate enough samples per data point that some can be withheld to detect overfitting.

2. Train a valuation network using the estimates as noisy labels (one of [these](https://github.com/iancovert/amortized-valuation/blob/main/experiments/train_cifar.py) [scripts](https://github.com/iancovert/amortized-valuation/blob/main/experiments/train_tabular.py)).

3. To verify how close the valuation network's estimates are to the ground truth, you can generate many Monte Carlo samples for a subset of the dataset, and then check the estimation accuracy using a script like [this](https://github.com/iancovert/amortized-valuation/blob/main/experiments/evaluate_accuracy_cifar.py).
