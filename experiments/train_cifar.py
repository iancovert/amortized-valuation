import os
import argparse
import numpy as np
from glob import glob

import torch
import torch.nn as nn
import lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar

from resnet_cifar import ResNet18
from adv import ValuationModel, MarginalContributionStackDataset
from utils import aggregate_estimator_results


# Set up argument parser.
parser = argparse.ArgumentParser()

# Dataset parameters.
parser.add_argument('--dataset', type=str, default='cifar10-10', choices=['cifar10-10', 'cifar10-25', 'cifar10-50'])
parser.add_argument('--num_points', type=int, default=50000)

# Data loader parameters.
parser.add_argument('--mbsize', type=int, default=256)
parser.add_argument('--train_contributions', type=int, default=5)
parser.add_argument('--val_contributions', type=int, default=1)
parser.add_argument('--target_scaling', type=float, default=0.001)
args = parser.parse_args()


# ------------------------------
# Experiment setup
# ------------------------------

# Parse arguments.
dataset = args.dataset
num_points = args.num_points


# ------------------------------
# Valuation model training
# ------------------------------

# Parse valuation model hyperparameters.
mbsize = args.mbsize
train_contributions = args.train_contributions
val_contributions = args.val_contributions
target_scaling = args.target_scaling

# Prepare standard training dataset.
train_dataset = torch.load(os.path.join('data_files', dataset, 'processed.pt'))['train_dataset']
train_dataset.transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_dataset = Subset(train_dataset, range(num_points))
val_dataset = Subset(
    torch.load(os.path.join('data_files', dataset, 'processed.pt'))['train_dataset'],
    range(num_points))

# Prepare Monte Carlo samples for training and validation.
filenames = np.sort(glob(os.path.join('data_files', dataset, 'mc-50000-50000-*.npy')))
results = [np.load(filename, allow_pickle=True).item() for filename in filenames]
train_results = aggregate_estimator_results(results[:train_contributions])
val_results = aggregate_estimator_results(results[-val_contributions:])

# Set up train dataloader.
train_delta = torch.tensor(train_results['estimates']).unsqueeze(1).float()[:num_points]
train_dataset = MarginalContributionStackDataset(train_dataset, train_delta)
train_loader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, drop_last=True, num_workers=0)

# Set up val dataloader.
val_delta = torch.tensor(val_results['estimates']).unsqueeze(1).float()[:num_points]
val_dataset = MarginalContributionStackDataset(val_dataset, val_delta)
val_loader = DataLoader(val_dataset, batch_size=mbsize, shuffle=False, drop_last=False, num_workers=0)


# ------------------------------
# Set up for hyperparameter grid
# ------------------------------

# For tracking best model.
best_loss = np.inf
best_checkpoint = None

for max_epochs in (10, 20, 30):
    for (lr, min_lr) in [(1e-3, 1e-5), (2e-4, 1e-5)]:
        # Set up valuation model with zero-init output layer.
        classifier = ResNet18()
        classifier.load_state_dict(torch.load(os.path.join('model_results', dataset, 'resnet18_pretrained.pt')))
        nn.init.zeros_(classifier.linear.weight)
        nn.init.zeros_(classifier.linear.bias)
        model = ValuationModel(
            model=classifier,
            target_scaling=target_scaling,
            lr=lr,
            min_lr=min_lr,
            save_architecture=False
        )

        # Train model.
        best_callback = ModelCheckpoint(
            save_top_k=1,
            monitor='val_loss',
            filename='{epoch}-{val_loss:.8f}',
            verbose=True
        )
        epoch_callback = ModelCheckpoint(
            every_n_epochs=1,
            filename='{epoch}'
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            precision='bf16-mixed',
            accelerator='gpu',
            devices=[0],
            log_every_n_steps=10,
            num_sanity_val_steps=0,
            callbacks=[TQDMProgressBar(refresh_rate=10), best_callback, epoch_callback]
        )
        trainer.fit(model, train_loader, val_loader)

        # Store results.
        loss = best_callback.best_model_score.item()
        checkpoint = best_callback.best_model_path
        if loss < best_loss:
            print(f'New best model: max_epochs = {max_epochs}, lr = {lr}, min_lr = {min_lr}')
            best_loss = loss
            best_checkpoint = checkpoint

# Save results.
results = {
    'best_loss': best_loss,
    'best_checkpoint': best_checkpoint
}
results_dir = os.path.join('model_results', dataset)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
torch.save(results, os.path.join(results_dir, f'amortized-{num_points}-{train_contributions}.pt'))
