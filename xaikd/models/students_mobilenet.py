from torchvision.models import (
    mobilenetv3,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

from torch import nn
from functools import partial

from . import MODEL_GENERATORS


def _student_s_trained(num_classes, class_indices) -> nn.Module:

    assert class_indices is not None

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    fc: nn.Linear = model.classifier[3]

    fc.weight = nn.Parameter(fc.weight[class_indices, :])
    fc.bias = nn.Parameter(fc.bias[class_indices])

    return model


def _student_s(num_classes, **kwargs) -> nn.Module:
    return mobilenet_v3_small(num_classes=num_classes)


def _student_xs(num_classes, **kwargs) -> nn.Module:
    dilation = 1

    width_mult = 1
    bneck_conf = partial(mobilenetv3.InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        mobilenetv3.InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )
    inverted_residual_setting = [
        # same conf as mobiletnet-s
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        # modified conf
        bneck_conf(40, 5, 120, 24, True, "HS", 1, 1),
        bneck_conf(24, 5, 144, 24, True, "HS", 1, 1),
        bneck_conf(24, 5, 288, 48, True, "HS", 2, dilation),  # C4
        bneck_conf(
            48,
            5,
            128,
            48,
            True,
            "HS",
            1,
            dilation,
        ),
        bneck_conf(
            48,
            5,
            128,
            48,
            True,
            "HS",
            1,
            dilation,
        ),
    ]
    last_channel = adjust_channels(256)  # C5

    return mobilenetv3._mobilenet_v3(
        inverted_residual_setting,
        last_channel,
        num_classes=num_classes,
        weights=None,
        progress=False,
        **kwargs,
    )


def _generate_model_function():

    MODEL_GENERATORS[f"student-mobilenets"] = _student_s
    MODEL_GENERATORS[f"student-mobilenetxs"] = _student_xs
    MODEL_GENERATORS[f"student-mobilenets-trained"] = _student_s_trained


_generate_model_function()
