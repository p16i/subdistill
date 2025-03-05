import typing

from functools import partial

import numpy as np

import torch
from torch.utils.data import Dataset, random_split


from xaikd import constants

from ..register import add_dataset_to_registry
from . import get_fineclass_names_indices_of_superclass

from .original import CIFAR100


class CIFAR100SuperclassVsOthers(CIFAR100):
    def __init__(self, super_class: str):
        super().__init__()

        _, self.selected_classes = get_fineclass_names_indices_of_superclass(
            superclass=super_class
        )

        self.num_classes = 1

    def transform_target(self, target: int) -> int:
        if target in self.selected_classes:
            return 1
        else:
            return 0

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:

        return super().create_subset(
            train_split=train_split, target_transform=self.transform_target
        )


class CIFAR100ValSplitSuperclassVsOthers(CIFAR100SuperclassVsOthers):
    seed = 1

    def create_subset(self, train_split=False, target_transform=None) -> Dataset:
        trng = torch.Generator()
        trng.manual_seed(self.seed)

        ds: CIFAR100 = super().create_subset(train_split=True)
        total_size = ds.data.shape[0]

        np.testing.assert_allclose(total_size, 500 * 100)
        subsets = random_split(
            # if `use-val-split=True, both training and testing sets
            # come from the training set.
            ds,
            [
                constants.TRAINING_VAL_SPLIT_RATIO,
                1 - constants.TRAINING_VAL_SPLIT_RATIO,
            ],
            generator=trng,
        )

        selected_subset = subsets[0] if train_split else subsets[1]

        subset_data_indices = selected_subset.indices

        # here, we override the original data
        ds.data = ds.data[subset_data_indices]
        ds.targets = np.array(ds.targets)[subset_data_indices].tolist()

        print(f"[train_split={train_split}] len(indices):={ds.data.shape[0]} ")

        expected_size = (
            constants.TRAINING_VAL_SPLIT_RATIO
            if train_split
            else 1 - constants.TRAINING_VAL_SPLIT_RATIO
        )
        np.testing.assert_allclose(
            ds.data.shape[0] / total_size, expected_size, atol=1e-3
        )

        return ds


def construct_variant_datasets():
    for super_class in constants.CIFAR100_SUPER_CLASSES:

        add_dataset_to_registry(
            f"cifar100-{super_class}-vs-others",
            partial(CIFAR100SuperclassVsOthers, super_class=super_class),
        )

        add_dataset_to_registry(
            f"cifar100val-{super_class}-vs-others",
            partial(CIFAR100ValSplitSuperclassVsOthers, super_class=super_class),
        )


construct_variant_datasets()
