import typing


from functools import partial

import numpy as np

import torch
from torch.utils.data import Dataset, random_split, Subset

from torchvision import datasets as tvd


from xaikd import constants
from xaikd.utils import spurious_feature_generator

from ..register import add_dataset_to_registry
from ..interface import WithValidationSetMixin

from . import IMAGENET_SUPERCLASS_MAPPING
from .subclasses import ImageNetSuperClass


class TorchVisionDatasetImageNetWithCopyrightFeatures(tvd.ImageNet):
    arr_data_spurious: typing.List[int]  # if 0 then not spurious
    slug = "spurious-copyright"

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

        assert spurious_type in [0, 1]

        if spurious_type == 1:
            sample = spurious_feature_generator.imagenet_copyright(sample, seed=index)

        sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageNetSuperclassWithCopyrightFeatures(
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
        return TorchVisionDatasetImageNetWithCopyrightFeatures

    def _construct_dataset(
        self, train_split: bool
    ) -> TorchVisionDatasetImageNetWithCopyrightFeatures:
        return super().create_subset(train_split)  # type: ignore

    def create_subset(self, train_split: bool, with_spurious: bool = True):
        ds = self._construct_dataset(train_split=train_split)

        assert isinstance(ds, TorchVisionDatasetImageNetWithCopyrightFeatures)

        rng = np.random.default_rng(seed=1)

        n = len(ds.targets)

        arr_data_spurious = np.zeros(n, dtype=int)

        if train_split and with_spurious:
            # we assume that the first class get contaminated
            cls_ix_with_spurious_feature = self._selected_classes[0]
            indices = (
                np.argwhere(np.array(ds.targets) == cls_ix_with_spurious_feature)
                .reshape(-1)
                .tolist()
            )

            total = int(np.floor(len(indices) * self.contamination_level))

            selected_indices = rng.permutation(indices)[:total]

            # with watermark
            arr_data_spurious[selected_indices] = 1

        ds.arr_data_spurious = arr_data_spurious.tolist()

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
        for contamination_level in [0.1, 0.25, 0.5, 0.75, 1.0]:
            sslug = "--".join(
                [
                    f"imagenet-{superclass}",
                    TorchVisionDatasetImageNetWithCopyrightFeatures.slug,
                    f"{contamination_level}",
                ]
            )
            add_dataset_to_registry(
                sslug,
                partial(
                    ImageNetSuperclassWithCopyrightFeatures,
                    contamination_level=contamination_level,
                    superclass=superclass,
                ),
            )


ano()
