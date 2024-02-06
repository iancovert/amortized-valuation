import torch
import numpy as np
from typing import Union
from scipy.stats import spearmanr


def generate_metrics(
    estimates: Union[np.ndarray, torch.Tensor],
    ground_truth: Union[np.ndarray, torch.Tensor],
    relevant_inds: Union[np.ndarray, torch.Tensor] = None,
):
    """
    Generate metrics comparing estimates to ground truth.

    Args:
      estimates: estimated data valuation scores.
      ground_truth: ground truth data valuation scores.
      relevant_inds: subset of indices for ground truth values.
    """
    # Select relevant indices.
    if relevant_inds is not None:
        estimates = estimates[relevant_inds]
        ground_truth = ground_truth[relevant_inds]

    # Calculate metrics.
    if isinstance(estimates, np.ndarray):
        error = np.mean((estimates - ground_truth) ** 2)
        expl_var = 1 - error / np.mean((ground_truth.mean() - ground_truth) ** 2)
        corr = np.corrcoef(estimates, ground_truth)[0, 1]
        spearman = spearmanr(estimates, ground_truth)[0]
        sign = np.mean(np.sign(estimates) == np.sign(ground_truth))
    else:
        error = torch.mean((estimates - ground_truth) ** 2).item()
        expl_var = (
            1 - error / torch.mean((ground_truth.mean() - ground_truth) ** 2).item()
        )
        corr = torch.corrcoef(torch.stack([estimates, ground_truth]))[0, 1].item()
        spearman = spearmanr(estimates.cpu().detach(), ground_truth.cpu().detach())[0]
        sign = torch.mean(
            (torch.sign(estimates) == torch.sign(ground_truth)).float()
        ).item()
    return {
        "error": error,
        "expl_var": expl_var,
        "corr": corr,
        "spearman": spearman,
        "sign_agreement": sign,
    }
