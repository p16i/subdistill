from enum import Enum

from torch import nn
import torchvision

from xaikd import models

ApproximatorMode = Enum(
    "ApproximatorMode",
    ["HOMOGENOUS", "HOMOGENOUS_LOWRANK_ADAPTER", "HOMOGENOUS_LOWRANK"],
)


def normalize_mode_name(mode: ApproximatorMode) -> str:
    return f"{mode}".split(".")[-1].lower()


def construct_approximator_for(
    model: nn.Module,
    layer: str,
    compression_rate: float,
    mode: ApproximatorMode,
):
    num_classes = getattr(model, "num_classes")
    d = models.get_layer_output_dimensions(model, layer)
    k = int(compression_rate * d)

    # this will be adaptered to different arch.
    backbone_approximator = get_approximator_for_resnet18(
        layer, output_dimensions=k, num_classes=num_classes
    )

    if mode == ApproximatorMode.HOMOGENOUS:
        assert compression_rate == 1.0, f"`{mode}` only work with `compression_rate=0`"

        last_module = nn.Identity()
    elif mode == ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER:
        last_module = nn.Conv2d(in_channels=k, out_channels=d, kernel_size=1)
    elif mode == ApproximatorMode.HOMOGENOUS_LOWRANK:
        last_module = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=1)

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
        # todo: check whether they use stride=2?
        stride=2,
        dilate=False,
    )
