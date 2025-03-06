import typing

from functools import partial

import numpy as np

import torch
from torch.utils.data import Dataset, random_split


from xaikd import constants

from ..register import add_dataset_to_registry
from . import get_fineclass_names_indices_of_superclass

from .original import CIFAR100Base


class CIFAR100SuperclassVsOthers(CIFAR100Base):

    def __init__(self, superclass: str):
        super().__init__()
        self._superclass = superclass
        _, self._selected_classes = get_fineclass_names_indices_of_superclass(
            superclass=superclass
        )

    @property
    def selected_classes(self):
        return self._selected_classes

    @property
    def num_classes(self):
        return 1

    @property
    def target_transform(self):
        def transform(target):
            if target in self.selected_classes:
                return 1
            else:
                return 0

        return transform

    def create_subset(
        self,
        train_split=False,
    ):

        return super().create_subset(
            train_split=train_split,
        )


class CIFAR100ValSplitSuperclassVsOthers(CIFAR100SuperclassVsOthers):
    seed = 1

    def create_subset(self, train_split=False, target_transform=None):
        trng = torch.Generator()
        trng.manual_seed(self.seed)

        ds = super().create_subset(train_split=True)
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
    for superclass in constants.CIFAR100_SUPER_CLASSES:

        add_dataset_to_registry(
            f"cifar100-{superclass}-vs-others",
            partial(CIFAR100SuperclassVsOthers, superclass=superclass),
        )

        add_dataset_to_registry(
            f"cifar100-valsplit-{superclass}-vs-others",
            partial(CIFAR100ValSplitSuperclassVsOthers, superclass=superclass),
        )


construct_variant_datasets()
