import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset, TensorDataset


class MarginalContributionDataset(TensorDataset):
    """
    Dataset for averaged marginal contributions.

    Args:
      x: input features.
      y: response variable.
      delta: estimates of data valuation scores.
    """

    def __init__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        delta: Union[np.ndarray, torch.Tensor],
    ):
        # Implement via TensorDataset.
        super().__init__(x, y, delta)


class MarginalContributionStackDataset(Dataset):
    """
    Dataset for averaged marginal contributions.

    Args:
      dataset: dataset of input features and response variable.
      delta: estimates of data valuation scores.
    """

    def __init__(self, dataset: Dataset, delta: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(dataset) == len(delta)
        self.dataset = dataset
        self.delta = TensorDataset(delta)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return tuple([*self.dataset[index], *self.delta[index]])
