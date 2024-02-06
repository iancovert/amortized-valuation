import os
import tqdm
import torch
import numpy as np
from typing import List, Union, Optional, ClassVar
from sklearn.utils import check_random_state
from opendataval.metrics import Metrics
from opendataval.model import ModelFactory
from opendataval.experiment import ExperimentMediator
from opendataval.dataloader import mix_labels, DataFetcher
from opendataval.dataval.margcontrib.sampler import Sampler


def get_experiment_mediator(dataset_name: str, num_points: int):
    """
    Helper function for preparing opendataval experiment mediator.

    Args:
      dataset_name:
      num_points:
    """
    if dataset_name in ("adult", "MiniBooNE"):
        # Set up fetcher.
        fetcher = DataFetcher.setup(
            dataset_name=dataset_name,
            cache_dir="data_files/",
            force_download=False,
            random_state=np.random.RandomState(0),
            train_count=num_points,
            valid_count=100,
            test_count=500,
            add_noise=mix_labels,
            noise_kwargs={"noise_rate": 0.2},
        )

    elif "cifar" in dataset_name:
        # Load pre-processed and split dataset.
        data_dict = torch.load(os.path.join("data_files", dataset_name, "processed.pt"))
        x_train = data_dict["x_train_embed"]
        y_train = data_dict["y_train"]
        x_val = data_dict["x_val_embed"]
        y_val = data_dict["y_val"]
        x_test = data_dict["x_test_embed"]
        y_test = data_dict["y_test"]

        # Restrict training examples.
        x_train = x_train[:num_points]
        y_train = y_train[:num_points]

        # Convert labels to one-hot.
        y_train = torch.nn.functional.one_hot(y_train)
        y_val = torch.nn.functional.one_hot(y_val)
        y_test = torch.nn.functional.one_hot(y_test)

        # Set up fetcher.
        fetcher = DataFetcher.from_data_splits(
            x_train.numpy(),
            y_train.numpy(),
            x_val.numpy(),
            y_val.numpy(),
            x_test.numpy(),
            y_test.numpy(),
            one_hot=True,
        )

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Set up model and train as sanity check.
    pred_model = ModelFactory(model_name="sklogreg", fetcher=fetcher)
    x_train, y_train, *_, x_test, y_test = fetcher.datapoints
    model = pred_model.clone()
    model.fit(x_train, y_train)
    perf = Metrics.ACCURACY(y_test, model.predict(x_test).cpu())
    print(f"Baseline model accuracy: {perf=}")

    # Return experiment mediator.
    return ExperimentMediator(fetcher=fetcher, pred_model=pred_model)


def aggregate_estimator_results(results: List[dict]):
    """
    Helper function to aggregate results saved from multiple Monte Carlo runs.

    Args:
      results: list of dictionaries containing Monte Carlo estimates, in the format used by
        monte_carlo.py and monte_carlo_distributional.py (keys are 'estimates',
        'total_contribution', 'total_count', 'relevant_inds')
    """
    # Verify relevant inds.
    relevant_inds = results[0]["relevant_inds"]
    assert all(
        [np.array_equal(relevant_inds, result["relevant_inds"]) for result in results]
    )

    # Aggregate results.
    total_contribution = sum([result["total_contribution"] for result in results])
    total_count = sum([result["total_count"] for result in results])
    estimates = total_contribution / total_count
    aggregated_results = {
        "estimates": estimates,
        "total_contribution": total_contribution,
        "total_count": total_count,
        "relevant_inds": relevant_inds,
    }
    return aggregated_results


class SubsetTMCSampler(Sampler):
    """Modified TMCSampler that operates on a subset of datapoints and does not perform truncation.

    Modeled on samplers from OpenDataVal:
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/margcontrib/sampler.py

    References
    ----------
    .. [1]  A. Ghorbani and J. Zou,
        Data Shapley: Equitable Valuation of Data for Machine Learning,
        arXiv.org, 2019. Available: https://arxiv.org/abs/1904.02868.

    .. [2]  Y. Kwon and J. Zou,
        Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for
        Machine Learning,
        arXiv.org, 2021. Available: https://arxiv.org/abs/2110.14049.

    Parameters
    ----------
    mc_epochs : int
        Number of outer epochs of MCMC sampling, by default 100
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    max_cardinality : int, optional
        Maximum cardinality of a training set, must be passed as kwarg, by default num_points
    cache_name : str, optional
        Unique cache_name of the model to  cache marginal contributions, set to None to
        disable caching, by default '' which is set to a unique value for a object
    random_state : np.random.RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    def __init__(
        self,
        mc_epochs: int = 100,
        min_cardinality: int = 5,
        max_cardinality: Union[int, None] = None,
        cache_name: Optional[str] = "",
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

        self.cache_name = None if cache_name is None else (cache_name or id(self))
        self.random_state = check_random_state(random_state)

    def set_coalition(self, coalition: torch.Tensor):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = len(coalition)
        self.total_contribution = np.zeros(self.num_points)
        self.total_count = np.zeros(self.num_points) + 1e-8

        # set max cardinality if not set
        if self.max_cardinality is None:
            self.max_cardinality = self.num_points
        return self

    def compute_marginal_contribution(
        self, relevant_inds: Union[np.ndarray, None] = None, *args, **kwargs
    ):
        """Compute the marginal contributions for semivalue based data evaluators.

        Parameters
        ----------
        relevant_inds : Union[np.ndarray, None], optional
            Indices of relevant data points, by default None
        args : tuple[Any], optional
             Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Checks cache if model name has been computed prior
        if self.cache_name is not None and self.cache_name in self.CACHE:
            return self.CACHE[self.cache_name]

        print("Start: marginal contribution computation", flush=True)

        for _ in tqdm.trange(self.mc_epochs):
            self._calculate_marginal_contributions(relevant_inds, *args, **kwargs)

        print("Done: marginal contribution computation", flush=True)

    def _calculate_marginal_contributions(
        self, relevant_inds: Union[np.ndarray, None] = None, *args, **kwargs
    ) -> np.ndarray:
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        relevant_inds : Union[np.ndarray, None], optional
            Indices of relevant data points, by default None
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        np.ndarray
            An array of marginal increments when one data point is added.
        """
        # Verify that relevant inds are valid.
        if relevant_inds is None:
            relevant_inds = set(np.arange(self.num_points))
        else:
            relevant_inds = set(relevant_inds)
            assert np.all(
                [((0 <= ind) and (ind < self.num_points)) for ind in relevant_inds]
            )

        # Generate random permutation.
        subset = self.random_state.permutation(self.num_points)
        coalition = list(subset[: self.min_cardinality])

        # Performance at minimal cardinality
        prev_perf = curr_perf = self.compute_utility(coalition, *args, **kwargs)

        for cutoff, idx in enumerate(
            subset[self.min_cardinality : self.max_cardinality],
            start=self.min_cardinality,
        ):
            # Add next member to coalition.
            coalition.append(idx)

            # Update utility if this index or next is relevant.
            if (idx in relevant_inds) or (
                (cutoff < self.num_points - 1) and (subset[cutoff + 1] in relevant_inds)
            ):
                curr_perf = self.compute_utility(coalition, *args, **kwargs)

            # Record marginal contribution if this index is relevant.
            if idx in relevant_inds:
                self.total_contribution[idx] += curr_perf - prev_perf
                self.total_count[idx] += 1

            # Update performance for next idx
            prev_perf = curr_perf


class SubsetDistributionalSampler(Sampler):
    """Distributional sampler that operates on a subset of datapoints.

    Modeled on samplers from OpenDataVal:
    https://github.com/opendataval/opendataval/blob/main/opendataval/dataval/margcontrib/sampler.py

    References
    ----------
    .. [1] A. Ghorbani et al.,
        A Distributional Framework for Data Valuation
        arXiv.org, 2020. Available: https://arxiv.org/abs/2002.12334.

    Parameters
    ----------
    mc_epochs : int
        Number of outer epochs of MCMC sampling, by default 100
    min_cardinality : int, optional
        Minimum cardinality of a training set, must be passed as kwarg, by default 5
    max_cardinality : int, optional
        Maximum cardinality of a training set, must be passed as kwarg, by default 1000
    cache_name : str, optional
        Unique cache_name of the model to  cache marginal contributions, set to None to
        disable caching, by default '' which is set to a unique value for a object
    random_state : RandomState, optional
        Random initial state, by default None
    """

    CACHE: ClassVar[dict[str, np.ndarray]] = {}
    """Cached marginal contributions."""

    def __init__(
        self,
        mc_epochs: int = 100,
        min_cardinality: int = 5,
        max_cardinality: int = 1000,
        cache_name: Optional[str] = "",
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.mc_epochs = mc_epochs
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

        self.cache_name = None if cache_name is None else (cache_name or id(self))
        self.random_state = check_random_state(random_state)

    def set_coalition(self, coalition: torch.Tensor):
        """Initializes storage to find marginal contribution of each data point"""
        self.num_points = len(coalition)
        self.total_contribution = np.zeros(self.num_points)
        self.total_count = np.zeros(self.num_points) + 1e-8
        return self

    def compute_marginal_contribution(
        self, relevant_inds: Union[np.ndarray, None] = None, *args, **kwargs
    ):
        """Compute the marginal contributions for semivalue based data evaluators.

        Parameters
        ----------
        relevant_inds : Union[np.ndarray, None], optional
            Indices of relevant data points, by default None
        args : tuple[Any], optional
             Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Checks cache if model name has been computed prior
        if self.cache_name is not None and self.cache_name in self.CACHE:
            return self.CACHE[self.cache_name]

        print("Start: marginal contribution computation", flush=True)

        for _ in tqdm.trange(self.mc_epochs):
            self._calculate_marginal_contributions(relevant_inds, *args, **kwargs)

        print("Done: marginal contribution computation", flush=True)

    def _calculate_marginal_contributions(
        self, relevant_inds: Union[np.ndarray, None] = None, *args, **kwargs
    ) -> np.ndarray:
        """Compute marginal contribution through TMC-Shapley algorithm.

        Parameters
        ----------
        relevant_inds : Union[np.ndarray, None], optional
            Indices of relevant data points, by default None
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments

        Returns
        -------
        np.ndarray
            An array of marginal increments when one data point is added.
        """
        # Verify that relevant inds are valid.
        if relevant_inds is None:
            relevant_inds = set(np.arange(self.num_points))
        else:
            relevant_inds = set(relevant_inds)
            assert np.all(
                [((0 <= ind) and (ind < self.num_points)) for ind in relevant_inds]
            )

        # Sample preceding subset at random for each index.
        for idx in relevant_inds:
            # Sample preceding subset.
            cardinality = self.random_state.randint(
                self.min_cardinality, self.max_cardinality
            )
            coalition = list(
                self.random_state.choice(
                    self.num_points, size=cardinality, replace=True
                )
            )

            # Calculate utility before and after adding idx.
            prev_perf = self.compute_utility(coalition, *args, **kwargs)
            curr_perf = self.compute_utility(coalition + [idx], *args, **kwargs)

            # Record marginal contribution.
            self.total_contribution[idx] += curr_perf - prev_perf
            self.total_count[idx] += 1
