from abc import ABC, abstractmethod
import os

from copy import deepcopy

import typing

from pathlib import Path


import numpy as np


import torch
from torch.utils.data import DataLoader, Subset, Dataset, random_split

from torchvision import datasets as tvd
from torchvision import transforms


from dataclasses import dataclass


from xaikd import constants

DATADIR = Path(os.getenv("DATASET_ROOT", "./datasets"))
TORCHVISION_DATASET_DOWNLOAD = bool(int(os.getenv("TORCHVISION_DATASET_DOWNLOAD", "0")))

if TORCHVISION_DATASET_DOWNLOAD:
    print(f"[warning!] TORCHVISION_DATASET_DOWNLOAD={TORCHVISION_DATASET_DOWNLOAD}")


def build_dataloader(
    dataset: Dataset,
    shuffle,
    num_workers=12,
    batch_size=constants.DEFAULT_BATCH_SIZE,
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


def subsample_dataset(dataset: tvd.VisionDataset, ratio: float, seed: int) -> Subset:
    assert 0 < ratio <= 1

    assert isinstance(dataset, tvd.VisionDataset)

    if ratio == 1:
        # remark: we simply return the original dataset but wrap it in Subset.
        return Subset(dataset=dataset, indices=list(range(len(dataset))))

    rng = torch.Generator()
    rng.manual_seed(seed)

    subset, _ = random_split(dataset, [ratio, 1 - ratio], rng)

    return subset


@dataclass
class DatasetConfiguration(ABC):
    @abstractmethod
    def create_subset(self, train_split: bool) -> tvd.VisionDataset:
        pass

    @property
    @abstractmethod
    def input_transformation(self) -> typing.Callable:
        pass

    @property
    @abstractmethod
    def input_training_transformation(self) -> typing.Callable:
        pass

    @property
    @abstractmethod
    def _normalizer(self) -> transforms.Normalize:
        pass

    @property
    @abstractmethod
    def selected_classes(self) -> typing.List[int]:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    def name(self) -> str:
        # this is during the construction
        assert hasattr(self, "__name")

        return getattr(self, "__name")

    @property
    @abstractmethod
    def dataclass(self) -> typing.Type[Dataset]:
        pass

    @property
    @abstractmethod
    def target_transform(self) -> typing.Union[typing.Callable, None]:
        pass

    @property
    def input_statistics(self) -> typing.Tuple[typing.List[float], typing.List[float]]:
        return (
            self._normalizer.mean,
            self._normalizer.std,
        )

    def __str__(self) -> str:
        return self.name


from .register import construct
from .interface import WithValidationSetMixin
from . import cifar100, imagenet, celeba


def construct_dataloaders(
    dataset: DatasetConfiguration,
    training_data_ratio: float,
    seed: int,
    training_batch_size: int,
    use_validation_set: bool,
) -> typing.Tuple[
    DataLoader[Subset[tvd.VisionDataset]],
    DataLoader[Subset[tvd.VisionDataset]],
    DataLoader[Subset[tvd.VisionDataset]],
    DataLoader[tvd.VisionDataset],
]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    if isinstance(dataset, WithValidationSetMixin):
        print("we use dataset with built-in validation set")
        assert (
            use_validation_set
        ), "Dataset provides validation set, so use_validation_set must be True"

        assert (
            training_data_ratio == 0.8
        ), "When using a dataset with built-in validation set, training_data_ratio must be 0.8"

        ds_train, ds_val = dataset.create_train_val_split(rng=rng)
    else:
        ds_train_raw = dataset.create_subset(train_split=True)

        if use_validation_set:
            ratio_train = np.min(
                [constants.TRAINING_VAL_SPLIT_RATIO, training_data_ratio]
            )
            ratio_val = 1 - constants.TRAINING_VAL_SPLIT_RATIO
            ratio_rest = 1 - (ratio_train + ratio_val)
            assert 0 <= ratio_rest <= 1

            print(
                f"[use_validation={use_validation_set}]: ratio_train={ratio_train:.4f}, ratio_val={ratio_val:.4f}"
            )

            ds_train, ds_val, _ = random_split(
                ds_train_raw,
                [ratio_train, ratio_val, ratio_rest],
                rng,
            )
        else:
            # we do this to make the type compatable
            ds_train, _ = random_split(
                ds_train_raw,
                [training_data_ratio, 1 - training_data_ratio],
                rng,
            )

            ds_val = dataset.create_subset(train_split=False)

    # remark: we have to do it this way because the current version of
    #  `contaminate_dataset` function only work with `Subset.
    ds_test = dataset.create_subset(train_split=False)

    # remark: we set shuffle=False here becaue it is only used to learn bases.
    dl_train = build_dataloader(ds_train, shuffle=False)

    dl_val = build_dataloader(
        ds_val,
        shuffle=False,
    )

    dl_test = build_dataloader(
        ds_test,
        shuffle=False,
    )

    print(f"==== Dataset Information [use_validation_set={use_validation_set}] ====")
    for label, ds in [("train", ds_train), ("val", ds_val), ("test", ds_test)]:
        print(f"> split={label:5s}: count={len(ds)}")

    ds_train_with_aug = deepcopy(ds_train)

    # We have to make sure that the `dataset` attribute is an actual dataset containing tranform.
    # This avoids having a nested chain of Subsets.
    assert hasattr(ds_train.dataset, "transform")
    assert isinstance(ds_train.dataset, (tvd.CIFAR100, tvd.ImageNet, tvd.CelebA))

    assert not ds_train_with_aug.dataset is None
    ds_train_with_aug.dataset.transform = dataset.input_training_transformation  # type: ignore

    # this loader is used in the distillation process.
    dl_train_with_aug = build_dataloader(
        ds_train_with_aug,
        shuffle=True,
        batch_size=training_batch_size,
        drop_last=True,
    )

    return dl_train, dl_train_with_aug, dl_val, dl_test
