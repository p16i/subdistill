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

from torchvision import datasets as tvd

from torchvision.models import ResNet18_Weights

from tqdm import tqdm

from xaikd import constants, utils


from torchvision.datasets.folder import default_loader


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
    num_workers=2,
    batch_size=64,
    drop_last=False,
) -> DataLoader:
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


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
    raise NotImplemented("Obsolete: use the `rng` version")

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


def selected_subset_samples_for_classes_with_seed(
    labels: npt.NDArray,
    subsampling_ratio: float,
    seed: int,
) -> npt.NDArray:
    selected = []

    assert 0 < subsampling_ratio <= 1.0

    rng = np.random.default_rng(seed=seed)

    unique_labels = np.unique(labels)

    for cix in unique_labels:
        indices = np.argwhere(labels == cix).reshape(-1)
        num_samples_for_classes = np.floor(indices.shape[0] * subsampling_ratio).astype(
            int
        )

        permuted_indices = rng.permutation(indices)

        selected.extend(permuted_indices[:num_samples_for_classes].tolist())

    return np.array(selected)


def subsample_dataset(dataset: Dataset, ratio: float, seed: int) -> Subset:
    assert 0 < ratio <= 1

    # todo: this might be different across torchvision dataset
    labels = dataset.targets

    indices = selected_subset_samples_for_classes_with_seed(
        labels, subsampling_ratio=ratio, seed=seed
    )

    return Subset(dataset, indices=indices.tolist())


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


# class TwoClassesDataset(DatasetConfiguration):
#     def __init__(
#         self,
#         base: DatasetConfiguration,
#         selected_classes: typing.List[int],
#         num_train_samples: typing.Union[None, int] = None,
#     ):
#         assert np.logical_and(
#             np.array(selected_classes) >= 0,
#             np.array(selected_classes) < base.num_classes,
#         ).all()

#         self.base = base
#         self.selected_classes = selected_classes

#         # remark: this might be a bit confusing
#         self.num_classes = base.num_classes

#         self._normalizer = self.base._normalizer
#         self.input_transformation = self.base.input_transformation
#         self.input_training_transformation = self.base.input_training_transformation

#         self.num_train_samples = num_train_samples

#     def create_subset(self, train_split=False) -> Dataset:
#         return self.base.create_subset(train_split=train_split)

#     def loader(
#         self,
#         batch_size=128,
#         num_workers=2,
#         train_split=False,
#         shuffle=False,
#     ):
#         ds = self.create_subset(train_split=train_split)
#         labels = ds.targets

#         if train_split and self.num_train_samples is not None:
#             selected_data_indices = selected_subset_samples_for_classes(
#                 np.array(labels),
#                 self.selected_classes,
#                 samples_per_class=self.num_train_samples,
#             )
#         else:
#             selected_data_indices = np.argwhere(
#                 np.isin(labels, self.selected_classes)
#             ).reshape(-1)

#         subset = Subset(ds, list(selected_data_indices))

#         return DataLoader(
#             subset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle
#         )


# @register_dataset("cifar10")
# @dataclass(init=False)
# class CIFAR10(DatasetConfiguration):
#     def __init__(self):
#         self.num_classes = 10

#         self._normalizer = transforms.Normalize(
#             mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
#         )

#         self.input_transformation = transforms.Compose(
#             [transforms.ToTensor(), self._normalizer]
#         )

#         # ref: https://github.com/zju-vipa/NetGraft/blob/main/utils/data.py#L35
#         self.input_training_transformation = transforms.Compose(
#             [
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 self._normalizer,
#             ]
#         )

#         self.dataclass = tvd.CIFAR10

#         self.root = DATADIR / "cifar10"

#     def create_subset(self, train_split=False) -> Dataset:
#         return self.dataclass(
#             root=self.root,
#             train=train_split,
#             transform=self.input_transformation,
#             download=TORCHVISION_DATASET_DOWNLOAD,
#         )


from . import cifar100


@register_dataset("imagenet")
@dataclass(init=False)
class ImageNet(DatasetConfiguration):
    selected_classes = list(range(1000))

    def __init__(self):
        # remark: we need to set this manually.

        self.num_classes = 1000

        self._normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L38
        self.input_transformation = ResNet18_Weights.IMAGENET1K_V1.transforms()

        # ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L238
        self.input_training_transformation = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalizer,
            ]
        )

        np.testing.assert_allclose(
            self.input_transformation.mean, self._normalizer.mean
        )

        self.dataclass = tvd.ImageNet
        self.root = DATADIR / "imagenet"

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:
        return self.dataclass(
            root=self.root,
            split="train" if train_split else "val",
            transform=self.input_transformation,
            target_transform=target_transform,
        )

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        return target


class ImageNetSuperClass(ImageNet):
    # remark: the targets are defined in the ImageNet dataset.
    selected_classes = []

    def __init__(self):
        super().__init__()

        self._target_mapping = dict(
            zip(self.selected_classes, range(len(self.selected_classes)))
        )

        self.num_classes = len(self.selected_classes)

    def create_subset(self, train_split=False) -> Dataset:
        ds = super().create_subset(
            train_split=train_split, target_transform=lambda t: self._target_mapping[t]
        )

        indices = np.argwhere(np.isin(ds.targets, self.selected_classes)).reshape(-1)

        print(f"We have {len(indices)} images in classes {self.selected_classes}")

        selected_samples = []
        selected_targets = []

        for six in tqdm(indices, desc=f"preparing `{self.__class__.__name__}` samples"):
            selected_samples.append(ds.samples[six])
            selected_targets.append(ds.targets[six])

        ds.imgs = selected_samples
        ds.samples = selected_samples

        ds.targets = selected_targets

        return ds


@register_dataset("imagenet-butterfly")
class ImageNetButterfly(ImageNetSuperClass):
    # remark: the targets are defined in the ImageNet dataset.
    selected_classes = [321, 322, 323, 324, 325, 326]


@register_dataset("imagenet-boat")
class ImageNetBoat(ImageNetSuperClass):
    # remark: the targets are defined in the ImageNet dataset.
    selected_classes = [472, 554, 576, 625, 814, 914]


@register_dataset("imagenet-car")
class ImageNetCar(ImageNetSuperClass):
    # remark: the targets are defined in the ImageNet dataset.
    selected_classes = [407, 436, 468, 511, 609, 627, 656, 661, 751, 817]


@register_dataset("imagenet-cat")
class ImageNetCat(ImageNetSuperClass):
    # remark: the targets are defined in the ImageNet dataset.
    selected_classes = [281, 282, 283, 284, 285, 286, 287]


@register_dataset("imagenet-edible_fruit")
class ImageNetEdibleFruit(ImageNetSuperClass):
    # remark: the targets are defined in the ImageNet dataset.
    selected_classes = [
        948,
        949,
        950,
        951,
        952,
        953,
        954,
        955,
        956,
        957,
    ]


@register_dataset("imagenet-fungus")
class ImageNetFungus(ImageNetSuperClass):
    selected_classes = [
        991,
        993,
        994,
        995,
        996,
        997,
    ]


@register_dataset("imagenet-truck")
class ImageNetTruck(ImageNetSuperClass):
    selected_classes = [
        555,
        569,
        656,
        675,
        717,
        734,
        864,
        867,
    ]


class ImageNetWithCopyRight(tvd.ImageNet):
    copyright = default_loader(
        str(
            Path(os.path.dirname(constants.PACKAGE_DIR))
            / "resources"
            / "copyright"
            / "1.png"
        )
    )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]

        sample = self.loader(path)

        assert self.transform is not None

        if index in self.victim_indices:
            sample = utils.apply_copyright_to_image(sample, self.copyright)

        sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


@register_dataset("imagenet-butterfly-spurious0.5")
class ImageNetButterflySpurrious(ImageNetButterfly):
    contamination_level = 0.5

    def __init__(self):
        super().__init__()

        self.dataclass = ImageNetWithCopyRight

    def create_subset(self, train_split=False) -> Dataset:
        ds = super().create_subset(train_split)

        rng = np.random.default_rng(seed=1)

        n = len(ds.targets)

        # todo: we fix this for now admiral
        victim_class = 321

        if train_split:
            indices = (
                np.argwhere(np.array(ds.targets) == victim_class).reshape(-1).tolist()
            )
        else:
            indices = list(range(n))

        total = int(n * self.contamination_level)

        selected_indices = rng.permutation(indices)[:total]
        ds.victim_indices = selected_indices

        return ds
