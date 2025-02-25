import typing

import numpy as np

import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import datasets as tvd


from torchvision.models import ResNet18_Weights


from . import DATADIR, DatasetConfiguration, register_dataset


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

    def transform_target(self, target: int) -> int:
        return target
