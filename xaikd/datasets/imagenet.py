import typing
import os

from dataclasses import dataclass

from functools import partial
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset, random_split

from torchvision import transforms
from torchvision import datasets as tvd

from torchvision.models import ResNet18_Weights
from torchvision.datasets.folder import default_loader

from tqdm import tqdm

from xaikd import constants, utils
from xaikd.utils import spurious_feature_generator

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

        if index in self.victim_indices:
            sample = self.modify_sample(sample)

        sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class TorchVisionDatasetImageNetWithCopyrightTag(
    TorchVisionDatasetImageNetWithSpuriousFeature
):
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

    def modify_sample(
        self, sample: spurious_feature_generator.TypeImage
    ) -> spurious_feature_generator.TypeImage:
        return spurious_feature_generator.imagenet_copyright(sample, self.copyright)


class TorchVisionDatasetImageNetWithWatermark(
    TorchVisionDatasetImageNetWithSpuriousFeature
):
    victim_indices: typing.List[int]

    slug = "spurious-watermark"

    def modify_sample(
        self, sample: spurious_feature_generator.TypeImage
    ) -> spurious_feature_generator.TypeImage:
        return spurious_feature_generator.imagenet_watermark(sample)


class TorchVisionDatasetImageNetWithJPEGArtifact(
    TorchVisionDatasetImageNetWithSpuriousFeature
):
    victim_indices: typing.List[int]

    slug = "spurious-jpeg"

    def modify_sample(
        self, sample: spurious_feature_generator.TypeImage
    ) -> spurious_feature_generator.TypeImage:
        return spurious_feature_generator.jpeg_artifact(sample)


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


class ImageNetSuperclasssValSplitWithSpurriousFeature(ImageNetSuperClass):
    seed = 1

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
        trng = torch.Generator()
        trng.manual_seed(self.seed)

        # always use training set
        ds = super().create_subset(train_split=True)

        ds: TorchVisionDatasetImageNetWithCopyrightTag

        subsets = random_split(
            # if `use-val-split=True, both training and testing sets
            # come from the training set.
            ds,
            [0.8, 0.2],
            generator=trng,
        )

        rng = np.random.default_rng(seed=1)

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

        if train_split:
            victim_class = np.min(self.selected_classes)
            # for `training` set,  we are only interested in only a class
            indices = (
                np.argwhere(np.array(ds.targets) == victim_class).reshape(-1).tolist()
            )
        else:
            # for `validation` set,  samples from all classes have the same
            # likelihood of having the spurious feature.
            indices = list(range(len(ds.targets)))

        total = int(np.floor(len(indices) * self.contamination_level))

        ds.victim_indices = rng.permutation(indices)[:total]

        return ds


class TorchVisionDatasetImageNetWithWatermarkJPEGTwoSpuriousFeatures(tvd.ImageNet):
    arr_data_spurious: typing.List[int]  # if 0 then not spurious
    slug = "spurious-watermarkjpeg"

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
        assert spurious_type in [0, 1, 2]

        if spurious_type == 1:
            sample = spurious_feature_generator.imagenet_watermark(sample)
        elif spurious_type == 2:
            sample = spurious_feature_generator.jpeg_artifact(sample)
        else:
            pass

        sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageNetSuperclasssWithWatermarkJPEGSpuriousFeatures(ImageNetSuperClass):

    def __init__(
        self,
        selected_classes: typing.List[int],
        contamination_level: float,
    ):
        super().__init__(selected_classes=selected_classes)

        self.contamination_level = contamination_level

        self.dataclass = TorchVisionDatasetImageNetWithWatermarkJPEGTwoSpuriousFeatures

    def create_subset(self, train_split=False) -> Dataset:
        ds = super().create_subset(train_split)

        ds: TorchVisionDatasetImageNetWithWatermarkJPEGTwoSpuriousFeatures

        rng = np.random.default_rng(seed=1)

        n = len(ds.targets)

        arr_data_spurious = np.zeros(n)

        sorted_classes = list(sorted(self.selected_classes))

        if train_split:
            victim_watermark_class = sorted_classes[0]
            victim_jpeg_class = sorted_classes[1]

            # for `training` set,  samples from the first two classes have spurious features:
            # - class 0: watermark
            # - class 1: jpeg
            for spurious_type_ix, cls_ix in [
                (1, victim_watermark_class),
                (2, victim_jpeg_class),
            ]:
                indices = (
                    np.argwhere(np.array(ds.targets) == cls_ix).reshape(-1).tolist()
                )

                total = int(np.floor(len(indices) * self.contamination_level))

                selected_indices = rng.permutation(indices)[:total]

                arr_data_spurious[selected_indices] = spurious_type_ix

        else:
            # for `validation` set,  samples from all classes have the same
            # likelihood to be either in these three groups
            # - clean (always 33%)
            # - spurious-watermark (contamination level * 33%)
            # - spurious-jpeg (contamintaion level * 33%)

            indices = list(range(n))

            total_per_type = int(
                np.floor(len(indices) * 0.33 * self.contamination_level)
            )

            permuted_indices = rng.permutation(indices)

            selected_indices_type_watermark = permuted_indices[:total_per_type]

            selected_indices_type_jpeg = permuted_indices[
                total_per_type : 2 * total_per_type
            ]

            assert (
                len(selected_indices_type_jpeg) - 1
                <= len(selected_indices_type_jpeg)
                <= len(selected_indices_type_jpeg) + 1
            )

            for spurious_type_ix, selected_indices in [
                (1, selected_indices_type_watermark),
                (2, selected_indices_type_jpeg),
            ]:
                arr_data_spurious[selected_indices] = spurious_type_ix

        ds.arr_data_spurious = arr_data_spurious.tolist()

        return ds


def ano():

    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        slug = f"imagenet-{superclass}"
        selected_classes = IMAGENET_SUPERCLASS_MAPPING[superclass]
        DATASETS[slug] = partial(ImageNetSuperClass, selected_classes=selected_classes)

        for dataclass in [
            TorchVisionDatasetImageNetWithCopyrightTag,
            TorchVisionDatasetImageNetWithWatermark,
            TorchVisionDatasetImageNetWithJPEGArtifact,
        ]:
            for contamination_level in [0.125, 0.25, 0.5, 1.0]:
                sslug = "--".join([slug, dataclass.slug, f"{contamination_level}"])
                DATASETS[sslug] = partial(
                    ImageNetSuperclasssWithSpurriousFeature,
                    contamination_level=contamination_level,
                    selected_classes=selected_classes,
                    dataclass=dataclass,
                )

    for dataclass in [
        TorchVisionDatasetImageNetWithCopyrightTag,
        TorchVisionDatasetImageNetWithWatermark,
        TorchVisionDatasetImageNetWithJPEGArtifact,
    ]:
        for contamination_level in [0.0, 0.5, 1.0]:
            sslug = "--".join(
                [
                    "imagenet-valsplit-butterfly",
                    dataclass.slug,
                    f"{contamination_level}",
                ]
            )
            DATASETS[sslug] = partial(
                ImageNetSuperclasssValSplitWithSpurriousFeature,
                contamination_level=contamination_level,
                selected_classes=IMAGENET_SUPERCLASS_MAPPING["butterfly"],
                dataclass=dataclass,
            )

        for contamination_level in [0.5, 1.0]:
            superclass = "butterfly"
            name = f"imagenet-{superclass}"
            spuroius_slug = (
                TorchVisionDatasetImageNetWithWatermarkJPEGTwoSpuriousFeatures.slug
            )
            sslug = "--".join([name, spuroius_slug, str(contamination_level)])
            DATASETS[sslug] = partial(
                ImageNetSuperclasssWithWatermarkJPEGSpuriousFeatures,
                selected_classes=IMAGENET_SUPERCLASS_MAPPING[superclass],
                contamination_level=contamination_level,
            )


ano()
