import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch import optim
from typing import Union
from .utils import generate_metrics
from sklearn.linear_model import LinearRegression


class ValuationModel(pl.LightningModule):
    """
    Amortized data valuation model. The model is trained to predict the
    expected marginal contribution of (x, y) pairs to a measure of the model
    quality (e.g., test set accuracy).

    Args:
      model: PyTorch network module.
      target_scaling: constant factor to scale predictions during training.
      ground_truth: ground truth valuation scores for validation.
      ground_truth_inds: subset of indices for ground truth validation.
      lr: learning rate.
      min_lr: minimum learning rate.
      save_architecture: save model architecture in hyperparameters.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_scaling: float = 1.0,
        ground_truth: Union[None, torch.Tensor] = None,
        ground_truth_inds: Union[None, torch.Tensor] = None,
        lr: float = 1e-3,
        min_lr: float = 1e-6,
        save_architecture: bool = True,
    ):
        # Store modules.
        super().__init__()
        self.model = model

        # Save ground truth.
        self.ground_truth = ground_truth
        self.ground_truth_inds = ground_truth_inds

        # Set optimization hyperparameters.
        self.target_scaling = target_scaling
        self.lr = lr
        self.min_lr = min_lr

        # Save hyperparameters.
        if save_architecture:
            self.save_hyperparameters()
        else:
            self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: Union[torch.Tensor, tuple[torch.Tensor]]):
        # Prepare input.
        if isinstance(batch, torch.Tensor):
            batch = (batch,)

        # Generate predictions for each class.
        x = batch[0]
        pred = self.model(x)

        # Rescale predictions if not training.
        if not self.training:
            pred = pred * self.target_scaling

        # Decode predictions for specified class.
        if len(batch) == 1:
            return pred
        else:
            y = batch[1]
            if len(y.shape) == 1:
                # Scalar encoding.
                return torch.gather(pred, 1, y.unsqueeze(1))
            else:
                # One-hot encoding.
                return torch.sum(pred * y, dim=1, keepdim=True)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Generate predictions.
        x, y, delta = batch
        pred = self((x, y))

        # Calculate loss.
        loss = nn.functional.mse_loss(pred, delta / self.target_scaling)
        return loss

    def on_validation_epoch_start(self):
        self.pred_list = []

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        # Generate predictions.
        x, y, delta = batch
        pred = self((x, y))

        # Log validation loss.
        loss = nn.functional.mse_loss(pred, delta)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(x))

        # Store results for later aggregation.
        self.pred_list.append(pred.cpu().float())

    def on_validation_epoch_end(self):
        # Compare aggregated predictions to ground truth.
        if self.ground_truth is not None:
            # Generate similarity metrics.
            estimates = torch.cat(self.pred_list).squeeze().cpu().float()
            metrics = generate_metrics(
                estimates, self.ground_truth, self.ground_truth_inds
            )
            self.log("error", metrics["error"], prog_bar=True)
            self.log("expl_var", metrics["expl_var"], prog_bar=True)
            self.log("corr", metrics["corr"], prog_bar=True)

            # Fit linear regression.
            linreg = LinearRegression().fit(
                estimates[self.ground_truth_inds].reshape(-1, 1).float().numpy(),
                self.ground_truth[self.ground_truth_inds].numpy(),
            )
            self.log("linreg_coef", linreg.coef_[0], prog_bar=True)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs, eta_min=self.min_lr, verbose=True
        )
        return {"optimizer": opt, "monitor": "val_loss", "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        # Save current estimates with checkpoint.
        checkpoint["estimates"] = torch.cat(self.pred_list).squeeze().cpu().float()

    def on_load_checkpoint(self, checkpoint):
        # Load estimates from checkpoint.
        self.estimates = checkpoint["estimates"]
