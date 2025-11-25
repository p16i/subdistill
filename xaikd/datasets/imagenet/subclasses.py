import typing
import os

from dataclasses import dataclass

from functools import partial
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, random_split, Subset

from torchvision import datasets as tvd

from torchvision.datasets.folder import default_loader

from tqdm import tqdm

from xaikd import constants
from xaikd.utils import spurious_feature_generator

from ..register import add_dataset_to_registry

from . import IMAGENET_SUPERCLASS_MAPPING
from .original import ImageNetBase


class ImageNetSuperClass(ImageNetBase):
    def __init__(
        self,
        superclass: str,
    ):
        super().__init__()

        self._superclass = superclass

        # remark: the targets are defined in the ImageNet dataset.
        self._selected_classes = IMAGENET_SUPERCLASS_MAPPING[superclass]

        self._target_mapping = dict(
            zip(self.selected_classes, range(len(self.selected_classes)))
        )

    def create_subset(self, train_split: bool):
        ds = super().create_subset(
            train_split=train_split,
        )

        indices = np.argwhere(np.isin(ds.targets, self.selected_classes)).reshape(-1)

        print(f"We have {len(indices)} images in classes {self.selected_classes}")

        selected_samples = []
        selected_targets = []

        for six in tqdm(
            indices,
            desc=f"Preparing `{self.__class__.__name__}[{self.selected_classes}]` samples",
        ):
            selected_samples.append(ds.samples[six])
            selected_targets.append(ds.targets[six])

        ds.imgs = selected_samples
        ds.samples = selected_samples

        ds.targets = selected_targets

        return ds

    @property
    def selected_classes(self):
        return self._selected_classes

    @property
    def num_classes(self):
        return len(self.selected_classes)

    @property
    def target_transform(self):
        def transform(target):
            return self._target_mapping[target]

        return transform


def ano():
    # construct watermark only
    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        slug = f"imagenet-{superclass}"
        add_dataset_to_registry(
            slug, partial(ImageNetSuperClass, superclass=superclass)
        )


ano()
