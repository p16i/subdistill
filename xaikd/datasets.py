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

from xaikd import constants


DATASETS = dict()

DATADIR = Path(os.getenv("DATASET_ROOT", "./datasets"))
TORCHVISION_DATASET_DOWNLOAD = int(os.getenv("TORCHVISION_DATASET_DOWNLOAD", "0"))
CIFAR100_SUPER_CLASSES = (
    pd.read_csv(constants.PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv")[
        "coarse_label_name"
    ]
    .unique()
    .tolist()
)

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
    dataset: Dataset, shuffle, num_workers=2, batch_size=64
) -> DataLoader:
    return DataLoader(
        dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle
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
    if name in DATASETS:
        dataset = DATASETS[name]()
    else:
        dataset_name, variant = _parse_dataset_name(name)
        if variant is not None:
            dataset_cls = DATASETS[dataset_name]
            # 55vs33
            match = re.match(r"(\d+)vs(\d+)", variant)
            if match:
                selected_classes = [int(match.group(1)), int(match.group(2))]
                dataset = TwoClassesDataset(dataset_cls(), selected_classes)
            elif dataset_name == "cifar100" and variant in CIFAR100_SUPER_CLASSES:
                dataset = Cifar100SuperClassesDataset(
                    dataset_cls(),
                    super_class=variant,
                )
        else:
            raise ValueError(f"{dataset_name} has no variant `{variant}` (name={name})")

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

        self._normalizer = self.base._normalizer
        self.input_transformation = self.base.input_transformation
        self.input_training_transformation = self.base.input_training_transformation

        self.num_train_samples = num_train_samples

    def create_subset(self, train_split=False) -> Dataset:
        return self.base.create_subset(train_split=train_split)

    def loader(
        self,
        batch_size=128,
        num_workers=2,
        train_split=False,
        shuffle=False,
    ):
        ds = self.create_subset(train_split=train_split)
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

        self._normalizer = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        )

        self.input_transformation = transforms.Compose(
            [transforms.ToTensor(), self._normalizer]
        )

        # ref: https://github.com/zju-vipa/NetGraft/blob/main/utils/data.py#L35
        self.input_training_transformation = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalizer,
            ]
        )

        self.dataclass = tvd.CIFAR10

        self.root = DATADIR / "cifar10"

    def create_subset(self, train_split=False) -> Dataset:
        return self.dataclass(
            root=self.root,
            train=train_split,
            transform=self.input_transformation,
            download=TORCHVISION_DATASET_DOWNLOAD,
        )


@register_dataset("cifar100")
@dataclass
class CIFAR100(DatasetConfiguration):
    selected_classes = list(range(100))

    def __init__(self):
        # ref: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py#L12C1-L13C83
        self._normalizer = transforms.Normalize(
            mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        )

        self.input_transformation = transforms.Compose(
            [transforms.ToTensor(), self._normalizer]
        )

        # ref: https://github.com/zju-vipa/NetGraft/blob/main/utils/data.py#L35
        self.input_training_transformation = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalizer,
            ]
        )

        self.num_classes = 100
        # remark: we use the transformation (Normalization) of CIFAR10 here!
        self.dataclass = tvd.CIFAR100
        self.root = DATADIR / "cifar100"

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:
        return self.dataclass(
            root=self.root,
            train=train_split,
            transform=self.input_transformation,
            download=TORCHVISION_DATASET_DOWNLOAD,
            target_transform=target_transform,
        )


@dataclass
class Cifar100SuperClassesDataset(DatasetConfiguration):
    def __init__(
        self,
        base: CIFAR100,
        super_class: str,
        num_train_samples: typing.Union[None, int] = None,
    ):
        # todo: refactor this not w.r.t the twoclass dataset
        self.base = base
        df_meta = pd.read_csv(
            constants.PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv"
        )

        df_selected = df_meta[df_meta.coarse_label_name == super_class]
        df_selected = df_selected.sort_values(by="fine_label")
        print(
            f"We are building `cifar100-{super_class}` containing {df_selected.shape[0]} fine classes"
        )
        for row in df_selected.to_dict("records"):
            print("> %s (%d)" % (row["fine_label_name"], row["fine_label"]))

        # remark: the targets are defined in the CIFAR100 dataset.
        self.selected_classes = df_selected.fine_label.values.tolist()

        self.num_classes = len(self.selected_classes)

        self._normalizer = self.base._normalizer
        self.input_transformation = self.base.input_transformation
        self.input_training_transformation = self.base.input_training_transformation

        # change name to mapping_old_and_new_target_indices
        # converting from old target (original dataset) to new target {0, 1,...})
        self._target_mapping = dict(
            zip(self.selected_classes, range(len(self.selected_classes)))
        )

    def create_subset(self, train_split=False) -> Dataset:
        ds = self.base.create_subset(
            train_split=train_split, target_transform=lambda t: self._target_mapping[t]
        )
        labels = ds.targets

        selected_data_indices = np.argwhere(
            np.isin(labels, self.selected_classes)
        ).reshape(-1)

        # here, we select samples belong to those targets.
        ds.data = ds.data[selected_data_indices, :]

        targets = np.array(ds.targets)[selected_data_indices].tolist()
        assert np.isin(targets, self.selected_classes).all()

        # remark: the targets here are still in the old system.
        # They will be converted to the new zero-indexing with target_transforms.
        # todo: add test
        #   comparing naive cifar100 and this dataset should have the same val
        ds.targets = targets

        return ds


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


@register_dataset("imagenet-butterfly")
class ImageNetButterfly(ImageNet):
    # remark: the targets are defined in the ImageNet dataset.
    selected_classes = [321, 322, 323, 324, 325, 326]

    def __init__(self):
        super().__init__()

        # todo: add unit tests  but mark.as.on server
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

        for six in tqdm(indices, desc="preparing `butterfly` samples"):
            selected_samples.append(ds.samples[six])
            selected_targets.append(ds.targets[six])

        ds.imgs = selected_samples
        ds.samples = selected_samples

        ds.targets = selected_targets

        return ds
