from enum import Enum
import numpy as np
from numpy import typing as npt

import torch
from torch import nn
import torchvision

from xaikd import models

ApproximatorMode = Enum(
    "ApproximatorMode",
    ["HOMOGENOUS", "HOMOGENOUS_LOWRANK_ADAPTER", "HOMOGENOUS_LOWRANK"],
)


def compute_compressed_dimension(d: int, compression_ratio: float) -> int:
    return int(np.floor(d / compression_ratio))


def normalize_mode_name(mode: ApproximatorMode) -> str:
    return f"{mode}".split(".")[-1].lower()


class Scale2D(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()

        # this mimics BatchNorm2d only its scaling functionality
        self.scale = nn.Parameter(torch.ones(d).reshape(1, d, 1, 1))

    def forward(self, x):
        return (self.scale**2) * x


def construct_approximator_for(
    model: nn.Module,
    layer: str,
    compression_ratio: float,
    mode: ApproximatorMode,
    seed: int,
):
    num_classes = getattr(model, "num_classes")
    d = models.get_layer_output_dimensions(model, layer)
    k = compute_compressed_dimension(d, compression_ratio)

    torch.manual_seed(seed)

    # todo: this will be adaptered to different arch.
    backbone_approximator = get_approximator_for_resnet18(
        layer, output_dimensions=k, num_classes=num_classes
    )

    if mode == ApproximatorMode.HOMOGENOUS:
        assert (
            compression_ratio == 1.0
        ), f"`{mode}` only work with `compression_rate=1.0`"

        last_module = Scale2D(d=d)
    elif mode == ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER:
        last_module = nn.Sequential(
            nn.Conv2d(in_channels=k, out_channels=d, kernel_size=1), Scale2D(d=d)
        )
    elif mode == ApproximatorMode.HOMOGENOUS_LOWRANK:
        last_module = Scale2D(d=k)

    return nn.Sequential(backbone_approximator, last_module)


def get_approximator_for_resnet18(
    layer: str, output_dimensions: int, num_classes=100
) -> nn.Module:
    model = models._resnet18_cifar(num_classes)
    model.inplanes = getattr(model, layer)[0].conv1.weight.shape[1]

    blocks = len(getattr(model, layer))

    return model._make_layer(
        torchvision.models.resnet.BasicBlock,
        output_dimensions,
        blocks,
        # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
        stride=2,
        # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
        dilate=False,
    )
