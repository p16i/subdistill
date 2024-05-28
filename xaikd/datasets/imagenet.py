import typing
import os

from dataclasses import dataclass

from functools import partial
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import datasets as tvd

from torchvision.models import ResNet18_Weights
from torchvision.datasets.folder import default_loader

from tqdm import tqdm

from xaikd import constants, utils

from . import DATASETS, DATADIR, register_dataset, DatasetConfiguration

DEFAULT_TRANSFORMATION = ResNet18_Weights.IMAGENET1K_V1.transforms()


@register_dataset("imagenet")
@dataclass(init=False)
class ImageNet(DatasetConfiguration):
    selected_classes = list(range(1000))

    def __init__(self):
        # remark: we need to set this manually.
        self.num_classes = 1000

        self._normalizer = transforms.Normalize(
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L44
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L38
        self.input_transformation = DEFAULT_TRANSFORMATION

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
    def __init__(self, selected_classes: typing.List[int]):
        super().__init__()

        # remark: the targets are defined in the ImageNet dataset.
        self.selected_classes = selected_classes

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


IMAGENET_SUPERCLASS_MAPPING = {
    "random": [100, 200, 300],  # for testing purpose
    "butterfly": [321, 322, 323, 324, 325, 326],
    "boat": [472, 554, 576, 625, 814, 914],
    "car": [407, 436, 468, 511, 609, 627, 656, 661, 751, 817],
    "cat": [281, 282, 283, 284, 285, 286, 287],
    "edible_fruit": [
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
    ],
    "fungus": [
        991,
        993,
        994,
        995,
        996,
        997,
    ],
    "truck": [
        555,
        569,
        656,
        675,
        717,
        734,
        864,
        867,
    ],
}


# @register_dataset("imagenet-butterfly")
# class ImageNetButterfly(ImageNetSuperClass):
#     # remark: the targets are defined in the ImageNet dataset.
#     selected_classes = [321, 322, 323, 324, 325, 326]


# @register_dataset("imagenet-boat")
# class ImageNetBoat(ImageNetSuperClass):
#     # remark: the targets are defined in the ImageNet dataset.
#     selected_classes = [472, 554, 576, 625, 814, 914]


# @register_dataset("imagenet-car")
# class ImageNetCar(ImageNetSuperClass):
#     # remark: the targets are defined in the ImageNet dataset.
#     selected_classes = [407, 436, 468, 511, 609, 627, 656, 661, 751, 817]


# @register_dataset("imagenet-cat")
# class ImageNetCat(ImageNetSuperClass):
#     # remark: the targets are defined in the ImageNet dataset.
#     selected_classes = [281, 282, 283, 284, 285, 286, 287]


# @register_dataset("imagenet-edible_fruit")
# class ImageNetEdibleFruit(ImageNetSuperClass):
#     # remark: the targets are defined in the ImageNet dataset.
#     selected_classes = [
#         948,
#         949,
#         950,
#         951,
#         952,
#         953,
#         954,
#         955,
#         956,
#         957,
#     ]


# @register_dataset("imagenet-fungus")
# class ImageNetFungus(ImageNetSuperClass):
#     selected_classes = [
#         991,
#         993,
#         994,
#         995,
#         996,
#         997,
#     ]


# @register_dataset("imagenet-truck")
# class ImageNetTruck(ImageNetSuperClass):
#     selected_classes = [
#         555,
#         569,
#         656,
#         675,
#         717,
#         734,
#         864,
#         867,
#     ]


class TorchVisionDatasetImageNetWithCopyrightTag(tvd.ImageNet):
    victim_indices: typing.List[int]

    slug = "spurious-copyright"

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


class ImageNetSuperclasssWithSpurriousFeature(ImageNetSuperClass):

    def __init__(
        self,
        selected_classes: typing.List[int],
        contamination_level: float,
        dataclass: typing.Type[tvd.ImageNet],
    ):
        super().__init__(selected_classes=selected_classes)

        self.contamination_level = contamination_level

        self.dataclass = dataclass

    def create_subset(self, train_split=False) -> Dataset:
        ds = super().create_subset(train_split)

        ds: TorchVisionDatasetImageNetWithCopyrightTag

        rng = np.random.default_rng(seed=1)

        n = len(ds.targets)

        if train_split:
            victim_class = np.min(self.selected_classes)
            # for `training` set,  we are only interested in only a class
            indices = (
                np.argwhere(np.array(ds.targets) == victim_class).reshape(-1).tolist()
            )
        else:
            # for `validation` set,  samples from all classes have the same
            # likelihood of having the spurious feature.
            indices = list(range(n))

        total = int(np.floor(len(indices) * self.contamination_level))

        ds.victim_indices = rng.permutation(indices)[:total]

        return ds


def ano():

    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        slug = f"imagenet-{superclass}"
        selected_classes = IMAGENET_SUPERCLASS_MAPPING[superclass]
        DATASETS[slug] = partial(ImageNetSuperClass, selected_classes=selected_classes)

        for dataclass in [TorchVisionDatasetImageNetWithCopyrightTag]:
            for contamination_level in [0.125, 0.25, 0.5, 1.0]:
                sslug = "--".join([slug, dataclass.slug, f"{contamination_level}"])
                DATASETS[sslug] = partial(
                    ImageNetSuperclasssWithSpurriousFeature,
                    contamination_level=contamination_level,
                    selected_classes=selected_classes,
                    dataclass=dataclass,
                )


ano()
