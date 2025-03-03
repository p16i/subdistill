import typing

import numpy as np

import torch

from functools import partial

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import datasets as tvd


from torchvision.models import ResNet18_Weights


from . import DATADIR, DatasetConfiguration, register_dataset, DATASETS


CELEBA_NUM_ATTRIBUTES = 40

DEFAULT_TRANSFORMATION = ResNet18_Weights.IMAGENET1K_V1.transforms()


@register_dataset("celeba")
class CelebA(DatasetConfiguration):
    def __init__(self):
        self._normalizer = transforms.Normalize(
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L44
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L38
        self.input_transformation = DEFAULT_TRANSFORMATION

        # ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L238
        # todo: when do we use this?
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

        self.dataclass = tvd.CelebA
        self.root = DATADIR

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:
        return self.dataclass(
            root=self.root,
            split="train" if train_split else "valid",
            transform=self.input_transformation,
            target_transform=target_transform,
        )

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        return target


class CelebAAttribute(CelebA):
    def __init__(self, attr_ix: int):
        super().__init__()

        assert 0 <= attr_ix < 40

        self.attr_ix = attr_ix

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:
        if target_transform is not None:
            print("Warning: setting target_transform has no effect here!")

        return self.dataclass(
            root=self.root,
            split="train" if train_split else "valid",
            transform=self.input_transformation,
            target_transform=self.transform_target,
        )

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:

        assert len(target.shape) == 1
        assert len(target) > 1

        return target[self.attr_ix]


def _register_celeba_attributes():
    for attr_ix in range(CELEBA_NUM_ATTRIBUTES):
        DATASETS[f"celeba-attr{attr_ix}"] = partial(CelebAAttribute, attr_ix=attr_ix)


_register_celeba_attributes()
