from abc import ABC, abstractmethod
import re
import os

import typing

from pathlib import Path

import pandas as pd

import numpy as np
import numpy.typing as npt


import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset, random_split

from torchvision import transforms


from dataclasses import dataclass


from tqdm import tqdm

from xaikd import constants, utils

from .multitask_mnist_fmnist import (
    MultiTaskMNISTFashionMNIST,
    MultiTaskEMNISTFashionMNIST,
)

# todo: write a function that add class to this dict with a pre-step that check name collistion
DATASETS = dict()

DATADIR = Path(os.getenv("DATASET_ROOT", "./datasets"))
TORCHVISION_DATASET_DOWNLOAD = int(os.getenv("TORCHVISION_DATASET_DOWNLOAD", "0"))

if TORCHVISION_DATASET_DOWNLOAD:
    print(f"[warning!] TORCHVISION_DATASET_DOWNLOAD={TORCHVISION_DATASET_DOWNLOAD}")


def register_dataset(name):
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        """Wrapped function to register a data modality provider with name `name`"""
        DATASETS[name] = cls

        return cls

    return wrapped


def build_dataloader(
    dataset: Dataset,
    shuffle,
    num_workers=12,
    batch_size=64,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
) -> DataLoader:

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def subsample_dataset(dataset: Dataset, ratio: float, seed: int) -> Subset:
    assert 0 < ratio <= 1

    if ratio == 1:
        # todo: add test for this
        # remark: we simply return the original dataset but wrap it in Subset.
        return Subset(dataset=dataset, indices=list(range(len(dataset))))

    rng = torch.Generator()
    rng.manual_seed(seed)

    subset, _ = random_split(dataset, [ratio, 1 - ratio], rng)

    return subset


@dataclass
class DatasetConfiguration(ABC):
    num_classes: int
    # input_statistics: typing.Tuple[typing.Tuple[float, ...], typing.Tuple[float, ...]]
    _normalizer: transforms.Normalize
    input_transformation: typing.Callable
    input_training_transformation: typing.Callable
    dataclass: typing.Callable
    selected_classes: typing.List[int]

    @abstractmethod
    def create_subset(self, train_split=False) -> Dataset:
        pass

    @property
    def input_statistics(self) -> typing.Tuple[typing.List[float], typing.List[float]]:
        return [
            self._normalizer.mean,
            self._normalizer.std,
        ]

    def __str__(self) -> str:
        return getattr(self, "__name")


def construct(name: str) -> DatasetConfiguration:
    assert name in DATASETS, f"dataset={name} does not exist!"

    dataset = DATASETS[name]()
    setattr(dataset, "__name", name)

    return dataset


from . import cifar100, imagenet, celeba
