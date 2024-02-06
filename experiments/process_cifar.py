import torch
import torch.nn as nn
import argparse
import numpy as np
import lightning as pl
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import ResNet50_Weights, resnet50


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--noise_portion", type=float, default=0.1)
args = parser.parse_args()

# Parse arguments.
noise_portion = args.noise_portion


class InferenceModule(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch[0])


# Prepare data for ResNet50 feature extraction.
image_transforms = transforms.Compose(
    [
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
train_dataset = CIFAR10(
    root="data_files/cifar10", transform=image_transforms, train=True, download=True
)
val_dataset = CIFAR10(
    root="data_files/cifar10", transform=image_transforms, train=False, download=True
)
test_dataset = CIFAR10(
    root="data_files/cifar10", transform=image_transforms, train=False, download=True
)

# Split smaller dataset into validation and test.
val_dataset.data = val_dataset.data[:1000]
val_dataset.targets = val_dataset.targets[:1000]
test_dataset.data = test_dataset.data[1000:10000]
test_dataset.targets = test_dataset.targets[1000:10000]
print(
    f"Train size = {len(train_dataset)}, Val size = {len(val_dataset)}, Test size = {len(test_dataset)}"
)

# Generate embeddings.
train_loader = DataLoader(train_dataset, batch_size=256, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, num_workers=0)
embedder = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
embedder.fc = nn.Identity()
inference_module = InferenceModule(embedder)
trainer = pl.Trainer(
    precision="bf16-mixed", accelerator="gpu", devices=[0], enable_progress_bar=False
)
x_train_embed = torch.cat(trainer.predict(inference_module, train_loader)).cpu().float()
x_val_embed = torch.cat(trainer.predict(inference_module, val_loader)).cpu().float()
x_test_embed = torch.cat(trainer.predict(inference_module, test_loader)).cpu().float()

# Reduce dimensionality of embeddings.
pca = PCA(n_components=256).fit(x_train_embed)
x_train_embed = torch.tensor(pca.transform(x_train_embed))
x_val_embed = torch.tensor(pca.transform(x_val_embed))
x_test_embed = torch.tensor(pca.transform(x_test_embed))

# Extract labels, add noise to training labels.
y_train = torch.tensor([y for _, y in train_dataset])
y_val = torch.tensor([y for _, y in val_dataset])
y_test = torch.tensor([y for _, y in test_dataset])
increment = torch.tensor(np.random.RandomState(0).randint(1, 10, size=len(y_train)))
num_preserve = int((1 - noise_portion) * len(y_train))
preserve_inds = np.random.RandomState(0).choice(
    len(y_train), size=num_preserve, replace=False
)
increment[preserve_inds] = 0
y_train_noise = (y_train + increment) % 10
train_dataset.targets = y_train_noise.tolist()

# Set transforms for CIFAR ResNet.
image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
train_dataset.transform = image_transforms
val_dataset.transform = image_transforms
test_dataset.transform = image_transforms

# Save dataset.
data_dict = {
    # Train.
    "train_dataset": train_dataset,
    "x_train_embed": x_train_embed,
    "y_train": y_train_noise,
    "y_train_true": y_train,
    # Val.
    "val_dataset": val_dataset,
    "x_val_embed": x_val_embed,
    "y_val": y_val,
    # Test.
    "test_dataset": test_dataset,
    "x_test_embed": x_test_embed,
    "y_test": y_test,
}
torch.save(data_dict, f"data_files/cifar10-{int(100 * noise_portion)}/processed.pt")
