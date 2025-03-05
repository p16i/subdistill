import typing

from functools import partial
import torch
from torch.utils.data import random_split


from tqdm import tqdm

from xaikd import constants

from ..register import add_dataset_to_registry
from . import IMAGENET_SUPERCLASS_MAPPING
from .original import ImageNetBase


class ImageNetSuperclassVsOthers(ImageNetBase):

    def __init__(self, super_class: str):
        super().__init__()

        self._superclass = super_class
        self._selected_classes = IMAGENET_SUPERCLASS_MAPPING[super_class]

    @property
    def target_transform(self):
        def transform(target):

            if target in self.selected_classes:
                return 1
            else:
                return 0

        return transform

    @property
    def selected_classes(self) -> typing.List[int]:
        return self._selected_classes

    @property
    def num_classes(self) -> int:
        return 1


class ImageNetValSplitSuperclassVsOthers(ImageNetSuperclassVsOthers):
    seed = 1

    def create_subset(self, train_split: bool):
        trng = torch.Generator()
        trng.manual_seed(self.seed)

        # always use training set
        ds = super().create_subset(train_split=True)

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
        print(
            f"[train_split={train_split}] len(subset_data_indices)={len(subset_data_indices)}"
        )

        arr_samples = []
        arr_targets = []

        for ix in tqdm(
            subset_data_indices,
            desc=f"Subseting train data (train_split={train_split}) or `{self.__class__.__name__}[{self.selected_classes}]` samples",
        ):
            arr_samples.append(ds.samples[ix])
            arr_targets.append(ds.targets[ix])

        # here, we override the original data
        ds.samples = arr_samples
        ds.imgs = arr_samples
        ds.targets = arr_targets

        return ds


def register_datasets():
    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        add_dataset_to_registry(
            f"imagenet-{superclass}-vs-others",
            partial(ImageNetSuperclassVsOthers, super_class=superclass),
        )

        add_dataset_to_registry(
            f"imagenet-valsplit-{superclass}-vs-others",
            partial(ImageNetValSplitSuperclassVsOthers, super_class=superclass),
        )


register_datasets()
