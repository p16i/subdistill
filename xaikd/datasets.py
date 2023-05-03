import typing

import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Subset

from torchvision import transforms


from dataclasses import dataclass

from torchvision import datasets as tvd


DATASETS = dict()


def register_dataset(name):
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        """Wrapped function to register a data modality provider with name `name`"""
        DATASETS[name] = cls

        return cls

    return wrapped


@dataclass
class DatasetConfiguration:
    num_classes: int
    input_normalization: typing.Callable
    transformation: typing.Callable
    dataclass: typing.Callable
    root: str = "./datasets"

    def loader(self, batch_size=64, num_workers=2, train_split=False):
        return DataLoader(
            self.dataclass(
                root=self.root,
                train=train_split,
                transform=self.transformation,
                download=True,
            ),
            num_workers=num_workers,
            batch_size=batch_size,
        )

    def __str__(self) -> str:
        return getattr(self, "__name")


def construct(name: str) -> DatasetConfiguration:
    slugs = name.split("-")

    assert len(slugs) in [1, 2]

    dataset = None

    if len(slugs) == 2:
        base, variant = slugs
        selected_classes = np.array(variant.split("vs")).astype(int)
        assert len(selected_classes) == 2

        base_dataset = construct(base)
        assert np.logical_and(
            selected_classes >= 0, selected_classes < base_dataset.num_classes
        ).all()

        dataset = TwoclassesDataset(base_dataset, selected_classes.tolist())
    else:
        dataset = DATASETS[name]()

    setattr(dataset, "__name", name)

    return dataset


class TwoclassesDataset(DatasetConfiguration):
    def __init__(self, base: DatasetConfiguration, selected_classes: typing.List[int]):
        self.base = base
        self.selected_classes = selected_classes

        # remark: this might be a bit confusing
        self.num_classes = base.num_classes

        self.input_normalization = self.base.input_normalization
        self.transformation = self.base.transformation

    def loader(self, batch_size=64, num_workers=2, train_split=False):
        ds = self.base.dataclass(
            root=self.base.root,
            train=train_split,
            transform=self.transformation,
            download=True,
        )

        selected_data_indices = np.argwhere(
            np.isin(ds.targets, self.selected_classes)
        ).reshape(-1)

        subset = Subset(ds, list(selected_data_indices))

        return DataLoader(
            subset,
            num_workers=num_workers,
            batch_size=batch_size,
        )


@register_dataset("cifar10")
@dataclass(init=False)
class CIFAR10(DatasetConfiguration):
    def __init__(self):
        self.num_classes = 10
        self.input_normalization = transforms.Normalize(
            [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        )

        self.transformation = transforms.Compose(
            [
                transforms.ToTensor(),
                self.input_normalization,
            ]
        )

        self.dataclass = tvd.CIFAR10


@register_dataset("cifar100")
@dataclass
class CIFAR100(CIFAR10):
    def __init__(self):
        super().__init__()

        self.num_classes = 100
        # remark: transformation (Normalization) of CIFAR100 should be different from CIFAR10
        self.dataclass = tvd.CIFAR100
