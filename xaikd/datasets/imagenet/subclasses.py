import typing
import os

from dataclasses import dataclass

from functools import partial
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, random_split

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


class TorchVisionDatasetImageNetWithSpuriousFeature(tvd.ImageNet):
    victim_indices: typing.List[int]
    slug: str

    def modify_sample(
        self, sample: spurious_feature_generator.TypeImage
    ) -> spurious_feature_generator.TypeImage:
        raise NotImplementedError("...")

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

        sample = spurious_feature_generator.scaling_artifact(sample)

        if index in self.victim_indices:
            sample = self.modify_sample(sample)

        sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


# class TorchVisionDatasetImageNetWithCopyrightTag(
#     TorchVisionDatasetImageNetWithSpuriousFeature
# ):
#     victim_indices: typing.List[int]

#     slug = "spurious-copyright"

#     copyright = default_loader(
#         str(
#             Path(os.path.dirname(constants.PACKAGE_DIR))
#             / "resources"
#             / "copyright"
#             / "1.png"
#         )
#     )

#     def modify_sample(
#         self, sample: spurious_feature_generator.TypeImage
#     ) -> spurious_feature_generator.TypeImage:
#         return spurious_feature_generator.imagenet_copyright(sample, self.copyright)


# class TorchVisionDatasetImageNetWithWatermark(
#     TorchVisionDatasetImageNetWithSpuriousFeature
# ):
#     victim_indices: typing.List[int]

#     slug = "spurious-watermark"

#     def modify_sample(
#         self, sample: spurious_feature_generator.TypeImage
#     ) -> spurious_feature_generator.TypeImage:
#         return spurious_feature_generator.imagenet_center_watermark(sample)


# class TorchVisionDatasetImageNetWithJPEGArtifact(
#     TorchVisionDatasetImageNetWithSpuriousFeature
# ):
#     victim_indices: typing.List[int]

#     slug = "spurious-jpeg"

#     def modify_sample(
#         self, sample: spurious_feature_generator.TypeImage
#     ) -> spurious_feature_generator.TypeImage:
#         return spurious_feature_generator.jpeg_artifact(sample)


# class ImageNetSuperclasssWithSpurriousFeature(ImageNetSuperClass):

#     def __init__(
#         self,
#         superclass: str,
#         contamination_level: float,
#         victim_class: int,
#         dataclass: typing.Type[tvd.ImageNet],
#     ):
#         super().__init__(superclass=superclass)

#         self.contamination_level = contamination_level

#         self._dataclass = dataclass
#         self.victim_class = victim_class

#     @property
#     def dataclass(self):
#         return self._dataclass

#     def create_subset(self, train_split=False):
#         ds = super().create_subset(train_split)

#         # ds: TorchVisionDatasetImageNetWithCopyrightTag

#         rng = np.random.default_rng(seed=1)

#         n = len(ds.targets)

#         if train_split:
#             victim_class = self.victim_class
#             # for `training` set,  we are only interested in only a class
#             indices = (
#                 np.argwhere(np.array(ds.targets) == victim_class).reshape(-1).tolist()
#             )
#         else:
#             # for `validation` set,  samples from all classes have the same
#             # likelihood of having the spurious feature.
#             indices = list(range(n))

#         total = int(np.floor(len(indices) * self.contamination_level))

#         ds.victim_indices = rng.permutation(indices)[:total]

#         return ds


# class ImageNetSuperclasssValSplitWithSpurriousFeature(ImageNetSuperClass):
#     seed = 1

#     def __init__(
#         self,
#         selected_classes: typing.List[int],
#         contamination_level: float,
#         dataclass: typing.Type[tvd.ImageNet],
#     ):
#         super().__init__(selected_classes=selected_classes)

#         self.contamination_level = contamination_level

#         self.dataclass = dataclass

#     def create_subset(self, train_split=False) -> Dataset:
#         trng = torch.Generator()
#         trng.manual_seed(self.seed)

#         # always use training set
#         ds = super().create_subset(train_split=True)

#         ds: TorchVisionDatasetImageNetWithCopyrightTag

#         subsets = random_split(
#             # if `use-val-split=True, both training and testing sets
#             # come from the training set.
#             ds,
#             [0.8, 0.2],
#             generator=trng,
#         )

#         rng = np.random.default_rng(seed=1)

#         selected_subset = subsets[0] if train_split else subsets[1]

#         subset_data_indices = selected_subset.indices

#         arr_samples = []
#         arr_targets = []

#         for ix in tqdm(
#             subset_data_indices,
#             desc=f"Subseting train data (train_split={train_split}) or `{self.__class__.__name__}[{self.selected_classes}]` samples",
#         ):
#             arr_samples.append(ds.samples[ix])
#             arr_targets.append(ds.targets[ix])

#         # here, we override the original data
#         ds.samples = arr_samples
#         ds.imgs = arr_samples
#         ds.targets = arr_targets

#         if train_split:
#             victim_class = np.min(self.selected_classes)
#             # for `training` set,  we are only interested in only a class
#             indices = (
#                 np.argwhere(np.array(ds.targets) == victim_class).reshape(-1).tolist()
#             )
#         else:
#             # for `validation` set,  samples from all classes have the same
#             # likelihood of having the spurious feature.
#             indices = list(range(len(ds.targets)))

#         total = int(np.floor(len(indices) * self.contamination_level))

#         ds.victim_indices = rng.permutation(indices)[:total]

#         return ds


class TorchVisionDatasetImageNetWithThreeSpuriousFeatures(tvd.ImageNet):
    total_spurious_types = 4
    arr_data_spurious: typing.List[int]  # if 0 then not spurious
    slug = "spurious-threespurious"

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

        spurious_type = self.arr_data_spurious[index]
        assert spurious_type in np.arange(
            self.total_spurious_types
        ), f"type={spurious_type}"

        if spurious_type == 1:
            sample = spurious_feature_generator.imagenet_center_watermark(sample)
        if spurious_type == 2:
            sample = spurious_feature_generator.imagenet_copyright(sample, seed=index)
        elif spurious_type == 3:
            sample = spurious_feature_generator.jpeg_artifact(sample)
        else:
            pass

        sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageNetSuperclassWithThreeSpuriousFeatures(ImageNetSuperClass):

    def __init__(
        self,
        superclass: str,
        contamination_level: float,
    ):
        super().__init__(superclass)

        assert 0 <= contamination_level <= 1.0

        self.contamination_level = contamination_level

    @property
    def dataclass(self):
        return TorchVisionDatasetImageNetWithThreeSpuriousFeatures

    def _construct_dataset(
        self, train_split: bool
    ) -> TorchVisionDatasetImageNetWithThreeSpuriousFeatures:
        return super().create_subset(train_split)

    def create_subset(self, train_split: bool):
        ds = self._construct_dataset(train_split=train_split)

        assert isinstance(ds, TorchVisionDatasetImageNetWithThreeSpuriousFeatures)

        rng = np.random.default_rng(seed=1)

        n = len(ds.targets)

        arr_data_spurious = np.zeros(n, dtype=int)

        sorted_classes = list(sorted(self.selected_classes))

        if train_split:
            # for `training` set,  samples from the first two classes have spurious features:
            # - spurious-watermark (type=1): class 1,5, ...
            # - spurious-copyright (type=2): class 2,6, ...
            # - spurious-jpeg      (type=3): class 3,7, ...
            for ix, cls_ix in enumerate(sorted_classes):
                spurious_type_ix = int(ix % self.dataclass.total_spurious_types)

                indices = (
                    np.argwhere(np.array(ds.targets) == cls_ix).reshape(-1).tolist()
                )

                total = int(np.floor(len(indices) * self.contamination_level))

                selected_indices = rng.permutation(indices)[:total]

                arr_data_spurious[selected_indices] = spurious_type_ix

        else:
            # for `validation` set,  samples from all classes have the same
            # likelihood to be either in these three groups
            # - type 1: spurious-watermark (contamination level * 100%)
            # - type 2: spurious-copyright (contamination level * 100%)
            # - type 3: spurious-jpeg (contamintaion level * 100 %)
            # - type 0: clean (the rest)

            indices = list(range(n))

            total_contaminated_samples = int(len(indices) * self.contamination_level)

            contaminated_samples = rng.permutation(indices)[:total_contaminated_samples]

            arr_spurious_types = np.arange(self.dataclass.total_spurious_types)

            assignment_data_spurious = rng.choice(
                arr_spurious_types, size=total_contaminated_samples
            )

            arr_data_spurious[contaminated_samples] = assignment_data_spurious

        ds.arr_data_spurious = arr_data_spurious.tolist()

        return ds


class ImageNetSuperclassValSplitWithThreeSpuriousFeatures(
    ImageNetSuperclassWithThreeSpuriousFeatures
):
    seed = 1

    def _construct_dataset(
        self, train_split: bool
    ) -> TorchVisionDatasetImageNetWithThreeSpuriousFeatures:
        trng = torch.Generator()
        trng.manual_seed(self.seed)

        # always use training set
        ds = super()._construct_dataset(train_split=True)

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


def ano():
    # construct watermark only
    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        slug = f"imagenet-{superclass}"
        add_dataset_to_registry(
            slug, partial(ImageNetSuperClass, superclass=superclass)
        )

    # construct threespurious
    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():

        for slug, arr_contamination_levels, dataset_class in [
            (
                superclass,
                [0.1, 0.25, 0.5, 0.75, 1.0],
                ImageNetSuperclassWithThreeSpuriousFeatures,
            ),
            (
                f"valsplit-{superclass}",
                [0.0, 1.0],
                ImageNetSuperclassValSplitWithThreeSpuriousFeatures,
            ),
        ]:

            for contamination_level in arr_contamination_levels:
                sslug = "--".join(
                    [
                        f"imagenet-{slug}",
                        TorchVisionDatasetImageNetWithThreeSpuriousFeatures.slug,
                        f"{contamination_level}",
                    ]
                )
                add_dataset_to_registry(
                    sslug,
                    partial(
                        dataset_class,
                        contamination_level=contamination_level,
                        superclass=superclass,
                    ),
                )


ano()
