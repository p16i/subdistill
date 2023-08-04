from enum import Enum
import numpy as np

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


def construct_approximator_for(
    model: nn.Module,
    layer: str,
    compression_ratio: float,
    mode: ApproximatorMode,
):
    num_classes = getattr(model, "num_classes")
    d = models.get_layer_output_dimensions(model, layer)
    k = compute_compressed_dimension(d, compression_ratio)

    # this will be adaptered to different arch.
    backbone_approximator = get_approximator_for_resnet18(
        layer, output_dimensions=k, num_classes=num_classes
    )

    if mode == ApproximatorMode.HOMOGENOUS:
        assert compression_ratio == 1.0, f"`{mode}` only work with `compression_rate=0`"

        last_module = nn.Identity()
    elif mode == ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER:
        last_module = nn.Conv2d(in_channels=k, out_channels=d, kernel_size=1)
    elif mode == ApproximatorMode.HOMOGENOUS_LOWRANK:
        last_module = nn.Sequential(
            nn.Conv2d(
                in_channels=k,
                out_channels=k,
                kernel_size=1,
            ),
            nn.BatchNorm2d(
                num_features=k,
                affine=False,
            ),
        )

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
