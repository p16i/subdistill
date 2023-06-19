from abc import ABC, abstractmethod
import re
import os

import typing

from pathlib import Path

import numpy as np
import numpy.typing as npt

from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset

from torchvision import transforms


from dataclasses import dataclass

from torchvision import datasets as tvd

from torchvision.models import ResNet18_Weights


DATASETS = dict()

DATADIR = Path("./datasets")
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


def _parse_dataset_name(
    name: str,
) -> typing.Tuple[str, typing.Union[None, str]]:
    # possible names:
    # cifar100, cifar100-5vs90

    slugs = name.split("-")

    variant = None

    if len(slugs) == 2:
        variant = slugs[-1]

    dataset_name = slugs[0]

    return dataset_name, variant


def selected_subset_samples_for_classes(
    labels: npt.NDArray,
    classes: typing.List[int],
    samples_per_class: int,
    verbose=False,
) -> npt.NDArray:
    selected = []

    assert set(classes).intersection(labels.tolist()) == set(classes)

    for cix in classes:
        indices = np.argwhere(labels == cix).reshape(-1)

        if indices.shape[0] < samples_per_class and verbose:
            print(
                f"[warning]: Class {cix} only has {indices.shape[0]} samples but we want {samples_per_class} samples!"
            )

        permuted_indices = np.random.permutation(indices)

        selected.extend(permuted_indices[:samples_per_class].tolist())

    return np.array(selected)


@dataclass
class DatasetConfiguration(ABC):
    num_classes: int
    input_statistics: typing.Tuple[typing.Tuple[float, ...], typing.Tuple[float, ...]]
    transformation: typing.Callable
    dataclass: typing.Callable

    @abstractmethod
    def create_dataset(slef, train_split=False) -> Dataset:
        pass

    def loader(self, batch_size=64, num_workers=2, train_split=False, shuffle=False):
        return DataLoader(
            self.create_dataset(train_split=train_split),
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def __str__(self) -> str:
        return getattr(self, "__name")


def construct(name: str, num_training_samples=None) -> DatasetConfiguration:
    dataset_name, variant = _parse_dataset_name(name)

    dataset_cls = DATASETS[dataset_name]

    if variant is not None:
        # 55vs33
        match = re.match(r"(\d+)vs(\d+)", variant)
        if match:
            selected_classes = [int(match.group(1)), int(match.group(2))]
            dataset = TwoClassesDataset(
                dataset_cls(), selected_classes, num_train_samples=num_training_samples
            )
        else:
            raise ValueError(f"{dataset_name} has no variant `{variant}`")
    else:
        if num_training_samples is not None:
            print(f"[warning: num_training_samples has NO effect on {name}")
        dataset = dataset_cls()

    setattr(dataset, "__name", name)

    return dataset


class TwoClassesDataset(DatasetConfiguration):
    def __init__(
        self,
        base: DatasetConfiguration,
        selected_classes: typing.List[int],
        num_train_samples: typing.Union[None, int] = None,
    ):
        assert np.logical_and(
            np.array(selected_classes) >= 0,
            np.array(selected_classes) < base.num_classes,
        ).all()

        self.base = base
        self.selected_classes = selected_classes

        # remark: this might be a bit confusing
        self.num_classes = base.num_classes

        self.input_statistics = self.base.input_statistics
        self.transformation = self.base.transformation

        self.num_train_samples = num_train_samples

    def create_dataset(self, train_split=False) -> Dataset:
        return self.base.create_dataset(train_split=train_split)

    def loader(
        self,
        batch_size=64,
        num_workers=2,
        train_split=False,
        shuffle=False,
        aug_transform=False,
    ):
        ds = self.create_dataset(train_split=train_split)
        if aug_transform:
            ds.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    self.base.transformation,
                ]
            )

        labels = ds.targets

        if train_split and self.num_train_samples is not None:
            selected_data_indices = selected_subset_samples_for_classes(
                np.array(labels),
                self.selected_classes,
                samples_per_class=self.num_train_samples,
            )
        else:
            selected_data_indices = np.argwhere(
                np.isin(labels, self.selected_classes)
            ).reshape(-1)

        subset = Subset(ds, list(selected_data_indices))

        return DataLoader(
            subset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle
        )


@register_dataset("cifar10")
@dataclass(init=False)
class CIFAR10(DatasetConfiguration):
    def __init__(self):
        self.num_classes = 10
        self.input_statistics = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        self.transformation = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*self.input_statistics)]
        )

        self.dataclass = tvd.CIFAR10

        self.root = DATADIR / "cifar10"

    def create_dataset(self, train_split=False) -> Dataset:
        return self.dataclass(
            root=self.root,
            train=train_split,
            transform=self.transformation,
            download=TORCHVISION_DATASET_DOWNLOAD,
        )


@register_dataset("cifar100")
@dataclass
class CIFAR100(CIFAR10):
    def __init__(self):
        super().__init__()

        self.num_classes = 100
        # remark: transformation (Normalization) of CIFAR100 should be different from CIFAR10
        self.dataclass = tvd.CIFAR100
        self.root = DATADIR / "cifar100"


@register_dataset("imagenet")
@dataclass(init=False)
class ImageNet(DatasetConfiguration):
    def __init__(self):
        self.num_classes = 1000
        # Ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L91
        self.input_statistics = (
            (0.43216, 0.394666, 0.37645),
            (0.22803, 0.22145, 0.216989),
        )

        self.transformation = ResNet18_Weights.IMAGENET1K_V1.transforms()

        self.dataclass = tvd.ImageNet
        self.root = DATADIR / "imagenet"

    def create_dataset(self, train_split=False) -> Dataset:
        return self.dataclass(
            root=self.root,
            split="train" if train_split else "val",
            transform=self.transformation,
        )
