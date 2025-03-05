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


from .register import register_dataset, add_dataset_to_registry
from . import DATADIR, DatasetConfiguration


NUM_CELEBA_ATTRIBUTES = 40

DEFAULT_TRANSFORMATION = ResNet18_Weights.IMAGENET1K_V1.transforms()


class CelebABase(DatasetConfiguration):

    def __init__(self):

        np.testing.assert_allclose(
            self.input_transformation.mean, self._normalizer.mean
        )

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:
        return self.dataclass(
            root=str(DATADIR),
            split="train" if train_split else "valid",
            transform=self.input_transformation,
            target_transform=self.target_transform,
        )

    @property
    def input_transformation(self):
        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L38
        return DEFAULT_TRANSFORMATION

    @property
    def input_training_transformation(self):
        # ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L238
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalizer,
            ]
        )

    @property
    def _normalizer(self) -> transforms.Normalize:

        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L44
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    @property
    def dataclass(self):
        return tvd.CelebA


@register_dataset("celeba")
class CelebA(CelebABase):

    @property
    def selected_classes(self) -> typing.List[int]:
        return list(range(NUM_CELEBA_ATTRIBUTES))

    @property
    def num_classes(self) -> int:
        return NUM_CELEBA_ATTRIBUTES

    @property
    def target_transform(self):
        return None


class CelebAAttribute(CelebABase):
    def __init__(self, attr_ix: int):
        super().__init__()

        assert 0 <= attr_ix < 40

        self.attr_ix = attr_ix

    @property
    def selected_classes(self) -> typing.List[int]:
        return [self.attr_ix]

    @property
    def num_classes(self) -> int:
        return 1

    @property
    def target_transform(self):
        def transform(target):

            assert len(target.shape) == 1
            assert len(target) > 1

            return target[self.attr_ix]

        return transform


def _register_celeba_attributes():
    for attr_ix in range(NUM_CELEBA_ATTRIBUTES):
        add_dataset_to_registry(
            f"celeba-attr{attr_ix}", partial(CelebAAttribute, attr_ix=attr_ix)
        )


_register_celeba_attributes()
