import typing

from torch import nn
from torch.utils.data import DataLoader

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


def get_constant(name) -> DatasetConfiguration:
    return DATASETS[name]()


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
