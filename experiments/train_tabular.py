import os
import argparse
import numpy as np
from glob import glob
from copy import deepcopy

import torch
import torch.nn as nn
import lightning as pl
from opendataval.dataval import DataShapley
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from adv import ValuationModel, Classifier, MarginalContributionDataset
from utils import get_experiment_mediator, aggregate_estimator_results


# Set up argument parser.
parser = argparse.ArgumentParser()

# Dataset parameters.
parser.add_argument("--dataset", type=str, default="adult")
parser.add_argument("--num_points", type=int, default=1000)

# Valuation model parameters.
parser.add_argument("--max_epochs", type=int, nargs="+", default=[10, 20, 30, 40, 50])
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--min_lr", type=float, default=5e-6)

# Data loader parameters.
parser.add_argument("--mbsize", type=int, default=32)
parser.add_argument("--train_contributions", type=int, default=10)
parser.add_argument("--val_contributions", type=int, default=10)
parser.add_argument("--target_scaling", type=float, default=0.001)
args = parser.parse_args()


# ------------------------------
# Experiment setup
# ------------------------------

# Parse arguments.
dataset = args.dataset
num_points = args.num_points

# Prepare training data.
exper_med = get_experiment_mediator(dataset, num_points)
data_evaluator = DataShapley(gr_threshold=1.05, max_mc_epochs=300, cache_name="cached")
data_evaluator.setup(exper_med.fetcher, exper_med.pred_model, exper_med.metric)
x_train = data_evaluator.x_train
y_train = data_evaluator.y_train


# ------------------------------
# Classifier pretraining
# ------------------------------

# Parse certain valuation model hyperparameters.
mbsize = args.mbsize

# Set up backbone.
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
classifier = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, output_dim),
)

# Set up dataloader.
classifier_dataset = TensorDataset(x_train, y_train.argmax(dim=1))
train_loader = DataLoader(
    classifier_dataset, batch_size=mbsize, shuffle=True, drop_last=True, num_workers=0
)
val_loader = DataLoader(
    classifier_dataset, batch_size=mbsize, shuffle=False, drop_last=False, num_workers=0
)

# Pre-train classifier.
model = Classifier(classifier, lr=1e-3)
trainer = pl.Trainer(
    max_epochs=20,
    precision="bf16-mixed",
    accelerator="gpu",
    devices=[0],
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    callbacks=[TQDMProgressBar(refresh_rate=10)],
)
trainer.fit(model, train_loader, val_loader)

# Verify accuracy.
preds = torch.cat(trainer.predict(model, val_loader))
acc = (preds.argmax(dim=1) == y_train.argmax(dim=1)).float().mean()
print(f"Classifier accuracy: {acc:.4f}")


# ------------------------------
# Valuation model training
# ------------------------------

# Parse valuation model hyperparameters.
lr = args.lr
min_lr = args.min_lr
mbsize = args.mbsize
train_contributions = args.train_contributions
val_contributions = args.val_contributions
target_scaling = args.target_scaling

# Prepare Monte Carlo samples for training and validation.
filenames = np.sort(
    glob(os.path.join("data_files", dataset, f"mc-{num_points}-{num_points}-*.npy"))
)
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
train_results = aggregate_estimator_results(results[:train_contributions])
val_results = aggregate_estimator_results(results[-val_contributions:])

# Set up train dataloader.
train_delta = torch.tensor(train_results["estimates"]).unsqueeze(1).float()
train_dataset = MarginalContributionDataset(x_train, y_train, train_delta)
train_loader = DataLoader(
    train_dataset, batch_size=mbsize, shuffle=True, drop_last=True, num_workers=0
)

# Set up val dataloader.
val_delta = torch.tensor(val_results["estimates"]).unsqueeze(1).float()
val_dataset = MarginalContributionDataset(x_train, y_train, val_delta)
val_loader = DataLoader(
    val_dataset, batch_size=mbsize, shuffle=False, drop_last=False, num_workers=0
)


# ------------------------------
# Set up for epochs grid
# ------------------------------

# For tracking best model.
best_loss = np.inf
best_checkpoint = None

for epochs in args.max_epochs:
    # Set up valuation model with zero-init output layer.
    classifier_copy = deepcopy(classifier)
    nn.init.zeros_(classifier_copy[-1].weight)
    nn.init.zeros_(classifier_copy[-1].bias)
    model = ValuationModel(
        model=classifier_copy, target_scaling=target_scaling, lr=lr, min_lr=min_lr
    )

    # Train model.
    best_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        filename="{epoch}-{val_loss:.8f}",
        verbose=True,
    )
    epoch_callback = ModelCheckpoint(every_n_epochs=1, filename="{epoch}")
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision="bf16-mixed",
        accelerator="gpu",
        devices=[0],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=10), best_callback, epoch_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    # Store results.
    loss = best_callback.best_model_score.item()
    checkpoint = best_callback.best_model_path
    if loss < best_loss:
        print(f"New best model: max_epochs = {epochs}")
        best_loss = loss
        best_checkpoint = checkpoint

# Save results.
results = {"best_loss": best_loss, "best_checkpoint": best_checkpoint}
results_dir = os.path.join("model_results", dataset)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
torch.save(
    results,
    os.path.join(results_dir, f"amortized-{num_points}-{train_contributions}.pt"),
)
