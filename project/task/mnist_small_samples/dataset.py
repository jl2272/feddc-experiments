"""MNIST dataset utilities for federated learning."""
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
from torchvision.datasets import MNIST

from project.task.default.dataset import (
    ClientDataloaderConfig as DefaultClientDataloaderConfig,
)
from project.task.default.dataset import (
    FedDataloaderConfig as DefaultFedDataloaderConfig,
)
from project.types.common import (
    CID,
    ClientDataloaderGen,
    FedDataloaderGen,
    IsolatedRNG,
)
from collections.abc import Sequence
from pathlib import Path
from typing import cast


import numpy as np
import torch
import random

from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import MNIST
from project.task.mnist_classification.dataset_preparation import _download_data, _balance_classes

# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig


def _partition_data(
        trainset: MNIST,
        testset: MNIST,
        num_clients: int,
        iid: bool,
        seed: int,
        balance: bool,
        num_samples: int,
) -> tuple[list[Subset] | list[ConcatDataset], MNIST]:
    """Split training set into iid or non iid partitions to simulate the federated.

    setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed
        by chunks to each client (used to test the convergence in a worst case scenario)
        , by default False
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default False
    num_samples : int
        The number of samples owned by each client.
    seed : int
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[MNIST], MNIST]
        A list of dataset for each client and a single dataset to be used for testing
        the model.
    """
    if balance:
        trainset = _balance_classes(trainset, seed)

    if len(trainset) < num_clients * num_samples:
        raise ValueError("There is not enough data for this number of samples per client")

    random.seed(seed)
    if iid:
        datasets = []
        random_indices = random.sample(range(len(trainset)), num_clients*num_samples)
        for i in range(num_clients):
            indices = random_indices[(i*num_samples):(i*num_samples)+num_samples]
            datasets.append(Subset(trainset, indices))
    else:
        shard_size = num_samples
        idxs = trainset.targets.argsort()
        sorted_data = Subset(
            trainset,
            cast(Sequence[int], idxs),
        )
        datasets = []

        max_start_shard = len(sorted_data) // shard_size
        start_positions = random.sample(range(max_start_shard), num_clients)
        for idx in range(num_clients):
            datasets.append(
                Subset(
                    sorted_data,
                    cast(
                        Sequence[int],
                        np.arange(
                            shard_size * start_positions[idx],
                            shard_size * (start_positions[idx] + 1),
                        ),
                    ),
                ),
            )
        random.shuffle(datasets)

    return datasets, testset


def get_dataloader_generators(
    dataset_dir: Path,
    num_clients: int = 25,
    iid: bool = False,
    seed: int = 42,
    balance: bool = False,
    num_samples: int = 64,
    val_ratio: float = 0.1
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    dataset_dir : Path
        Path of the dataset directory
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed
        by chunks to each client (used to test the convergence in a worst case scenario)
        , by default False
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default False
    num_samples : int
        The number of samples owned by each client. Default is 64
    seed : int
        Used to set a fix seed to replicate experiments, by default 42
    val_ratio : float
        Used to decide how many samples for validation. Defaults to 0.1

    Returns
    -------
    Tuple[ClientDataloaderGen, FedDataloaderGen]
        A tuple of functions that return a DataLoader for a client's dataset
        and a DataLoader for the federated dataset.
    """
    trainset, testset = _download_data(
        Path(dataset_dir),
    )

    # Partition the dataset
    # ideally, the fed_test_set can be composed in three ways:
    # 1. fed_test_set = centralized test set like MNIST
    # 2. fed_test_set = concatenation of all test sets of all clients
    # 3. fed_test_set = test sets of reserved unseen clients
    client_datasets, fed_test_set = _partition_data(
        trainset,
        testset,
        num_clients,
        iid,
        seed,
        balance,
        num_samples,
    )

    client_train = []
    client_test = []
    for client_dataset in client_datasets:
        len_val = int(
            len(client_dataset) * val_ratio,
        )
        lengths = [len(client_dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            client_dataset,
            lengths,
            torch.Generator().manual_seed(seed),
        )
        client_train.append(ds_train)
        client_test.append(ds_val)
    def get_client_dataloader(
        cid: CID, test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for a client's dataset.

        Parameters
        ----------
        cid : str|int
            The client's ID
        test : bool
            Whether to load the test set or not
        _config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
        DataLoader
            The DataLoader for the client's dataset
        """
        config: ClientDataloaderConfig = ClientDataloaderConfig(**_config)
        del _config

        torch_cpu_generator = rng_tuple[3]

        if test:
            dataset = client_test[int(cid)]
        else:
            dataset = client_train[int(cid)]
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    def get_federated_dataloader(
        test: bool, _config: dict, rng_tuple: IsolatedRNG
    ) -> DataLoader:
        """Return a DataLoader for federated train/test sets.

        Parameters
        ----------
        test : bool
            Whether to load the test set or not
        config : Dict
            The configuration for the dataset
        rng_tuple : IsolatedRNGTuple
            The random number generator state for the training.
            Use if you need seeded random behavior

        Returns
        -------
            DataLoader
            The DataLoader for the federated dataset
        """
        if not test:
            return None
        config: FedDataloaderConfig = FedDataloaderConfig(
            **_config,
        )
        del _config
        torch_cpu_generator = rng_tuple[3]

        return DataLoader(
            testset,
            batch_size=config.batch_size,
            shuffle=not test,
            generator=torch_cpu_generator,
        )

    return get_client_dataloader, get_federated_dataloader
