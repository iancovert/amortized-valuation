import torch
import argparse
import lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import TQDMProgressBar

from adv import Classifier
from resnet_cifar import ResNet18


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--noise_portion', type=float, default=0.1)
parser.add_argument('--max_epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--mbsize', type=int, default=32)
args = parser.parse_args()

# Parse arguments
noise_portion = args.noise_portion
max_epochs = args.max_epochs
mbsize = args.mbsize
lr = args.lr
min_lr = args.min_lr

# Load datasets.
data_dict = torch.load(f'data_files/cifar10-{int(100 * noise_portion)}/processed.pt')
train_dataset = data_dict['train_dataset']
val_dataset = data_dict['val_dataset']
test_dataset = data_dict['test_dataset']

# Add augmentations for training dataset.
train_dataset.transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Set up data loaders.
train_loader = DataLoader(train_dataset, batch_size=mbsize, shuffle=True, drop_last=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=mbsize, shuffle=False, drop_last=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=mbsize, shuffle=False, drop_last=False, num_workers=0)

# Train model.
model = ResNet18()
classifier = Classifier(model, lr=lr, min_lr=min_lr)
trainer = pl.Trainer(
    max_epochs=max_epochs,
    precision='bf16-mixed',
    accelerator='gpu',
    devices=[0],
    log_every_n_steps=10,
    num_sanity_val_steps=0,
    callbacks=[TQDMProgressBar(refresh_rate=10)]
)
trainer.fit(classifier, train_loader, val_loader)

# Calculate test set accuracy.
test_performance = trainer.test(model=classifier, dataloaders=test_loader)[0]
print(f'Test set accuracy: {test_performance["test_acc"]:.4f}')
print(f'Test set loss: {test_performance["test_loss"]:.4f}')

# Save final model.
torch.save(model.state_dict(), f'model_results/cifar10-{int(100 * noise_portion)}/resnet18_pretrained.pt')
