import typing
import numpy as np

from torchvision import transforms
from torchvision import datasets as tvd

from torchvision.models import ResNet18_Weights

from torchvision.transforms.functional import InterpolationMode

import torch

from ..register import register_dataset
from .. import DATADIR, DatasetConfiguration

DEFAULT_TRANSFORMATION = ResNet18_Weights.IMAGENET1K_V1.transforms()


class ImageNetBase(DatasetConfiguration):
    @property
    def input_transformation(self):
        # ref: see notebooks/2025-09-v0.8.x/dev/imagenet-transform.ipynb
        return transforms.Compose(
            [
                transforms.Resize(
                    size=256, interpolation=InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.CenterCrop(224),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                self._normalizer,
            ]
        )

    @property
    def input_training_transformation(self):
        # ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L238
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, interpolation=InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.RandomHorizontalFlip(),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                self._normalizer,
            ]
        )

    @property
    def _normalizer(self) -> transforms.Normalize:
        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L44
        return transforms.Normalize(
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L44
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def create_subset(
        self,
        train_split: bool,
    ) -> tvd.ImageNet:
        return self.dataclass(
            root=str(DATADIR / "imagenet"),
            split="train" if train_split else "val",
            transform=self.input_transformation,
            target_transform=self.target_transform,
        )

    @property
    def dataclass(self):
        return tvd.ImageNet


@register_dataset("imagenet")
class ImageNet(ImageNetBase):
    def __init__(self):
        np.testing.assert_allclose(
            ResNet18_Weights.IMAGENET1K_V1.transforms().mean, self._normalizer.mean
        )

    def transform_target(self, target: int) -> int:
        return target

    @property
    def selected_classes(self) -> typing.List[int]:
        return list(range(1000))

    @property
    def num_classes(self):
        return 1000

    @property
    def target_transform(self):
        return None
