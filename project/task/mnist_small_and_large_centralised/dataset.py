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
from project.task.mnist_small_and_large.dataset import _partition_data
# Use defaults for this very simple dataset
# Requires only batch size
ClientDataloaderConfig = DefaultClientDataloaderConfig
FedDataloaderConfig = DefaultFedDataloaderConfig

def get_dataloader_generators(
    dataset_dir: Path,
    num_clients: int = 25,
    seed: int = 42,
    val_ratio: float = 0.1,
    small_size: int = 16,
    large_size: int = 1200
) -> tuple[ClientDataloaderGen, FedDataloaderGen]:
    """Return a function that loads a client's dataset.

    Parameters
    ----------
    dataset_dir : Path
        Path of the dataset directory
    num_clients : int
        The number of clients that hold a part of the data
    seed : int
        Used to set a fix seed to replicate experiments, by default 42
    val_ratio : float
        Used to decide how many samples for validation. Defaults to 0.1
    small_size : int
        Number of samples for small clients
    large_size : int
        Number of samples for large clients

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
        seed,
        small_size,
        large_size
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
    client_train = ConcatDataset(client_train)
    client_test = ConcatDataset(client_test)

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
            dataset = client_test
        else:
            dataset = client_train
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
