import typing


from functools import partial

import numpy as np

import torch
from torch.utils.data import Dataset, random_split, Subset

from torchvision import datasets as tvd
from torchvision.transforms import functional as TF

from PIL import Image


from xaikd import constants
from xaikd.utils import spurious_feature_generator

from ..register import add_dataset_to_registry
from ..interface import WithValidationSetMixin

from . import IMAGENET_SUPERCLASS_MAPPING
from .subclasses import ImageNetSuperClass


class TorchVisionDatasetImageNetWithMNISTSpuriousFeatures(tvd.ImageNet):
    arr_index_with_spurious: typing.List[int]  # if 0 then not spurious

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        sample = self.loader(path)

        assert self.transform is not None

        sample = self.transform(sample)
        if index in self.arr_index_with_spurious:
            sample = TF.to_pil_image(sample)
            sample = spurious_feature_generator.mnist_corner(
                sample,
                label=target,
                seed=index,
            )
            sample = TF.to_tensor(sample)

        return sample, target


class ImageNetSuperclassWithMNISTSpuriousFeatures(
    ImageNetSuperClass, WithValidationSetMixin
):
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
        return TorchVisionDatasetImageNetWithMNISTSpuriousFeatures

    def _construct_dataset(
        self, train_split: bool
    ) -> TorchVisionDatasetImageNetWithMNISTSpuriousFeatures:
        return super().create_subset(train_split)

    def create_subset(self, train_split: bool, with_spurious: bool = True):
        ds = self._construct_dataset(train_split=train_split)

        assert isinstance(ds, TorchVisionDatasetImageNetWithMNISTSpuriousFeatures)

        rng = np.random.default_rng(seed=1)

        n = len(ds.targets)

        if train_split and with_spurious:
            total_spurios = int(n * self.contamination_level)
            print(f"We have {total_spurios} spurious samples out of {n} total samples.")
            arr_index_with_spurious = rng.permutation(n)[:total_spurios].tolist()
        else:
            arr_index_with_spurious = []

        ds.arr_index_with_spurious = arr_index_with_spurious

        return ds

    def create_train_val_split(
        self,
        training_size: float,
        rng: torch.Generator,
    ) -> typing.Tuple[Subset[tvd.VisionDataset], Subset[tvd.VisionDataset]]:
        ds_train_raw = self.create_subset(train_split=True)

        ratio_train = np.min([constants.TRAINING_VAL_SPLIT_RATIO, training_size])
        ratio_val = 1 - constants.TRAINING_VAL_SPLIT_RATIO
        ratio_rest = 1 - (ratio_train + ratio_val)
        assert 0 <= ratio_rest <= 1

        print(
            f"[{self.__class__.__name__} ratio_train={ratio_train:.4f}, ratio_val={ratio_val:.4f}"
        )

        ds_train, ds_val, _ = random_split(
            ds_train_raw,
            [ratio_train, ratio_val, ratio_rest],
            rng,
        )
        # make sure that we don't have any spurious data in val set
        ds_val.dataset = self.create_subset(train_split=True, with_spurious=False)

        return ds_train, ds_val


def ano():
    # construct copyright
    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        for contamination_level in [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]:
            sslug = "--".join(
                [
                    f"imagenet-{superclass}",
                    "mnistspurious",
                    f"{contamination_level}",
                ]
            )
            add_dataset_to_registry(
                sslug,
                partial(
                    ImageNetSuperclassWithMNISTSpuriousFeatures,
                    contamination_level=contamination_level,
                    superclass=superclass,
                ),
            )


ano()
