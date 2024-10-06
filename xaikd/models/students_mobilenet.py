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


def _student_very_small(num_classes, dim1, dim2, dim3, **kwargs) -> nn.Module:
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
        bneck_conf(40, 5, 120, dim1, True, "HS", 1, 1),
        bneck_conf(dim1, 5, dim2, dim1, True, "HS", 1, 1),
        bneck_conf(dim1, 5, dim2 * 2, dim1 * 2, True, "HS", 2, dilation),  # C4
        bneck_conf(
            dim1 * 2,
            5,
            dim3,
            dim1 * 2,
            True,
            "HS",
            1,
            dilation,
        ),
        bneck_conf(
            dim1 * 2,
            5,
            dim3,
            dim1 * 2,
            True,
            "HS",
            1,
            dilation,
        ),
    ]
    last_channel = adjust_channels(dim3 * 2)  # C5

    return mobilenetv3._mobilenet_v3(
        inverted_residual_setting,
        last_channel,
        num_classes=num_classes,
        weights=None,
        progress=False,
        **kwargs,
    )


def _student_very_small_cifar(num_classes, dim1, dim2, dim3, **kwargs) -> nn.Module:
    dilation = 1

    width_mult = 1
    bneck_conf = partial(mobilenetv3.InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        mobilenetv3.InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    inverted_residual_setting = [
        # same conf as mobiletnet-s
        bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 1, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        # modified conf
        bneck_conf(40, 5, 120, dim1, True, "HS", 1, 1),
        bneck_conf(dim1, 5, dim2, dim1, True, "HS", 1, 1),
        bneck_conf(dim1, 5, dim2 * 2, dim1 * 2, True, "HS", 2, dilation),  # C4
        bneck_conf(
            dim1 * 2,
            5,
            dim3,
            dim1 * 2,
            True,
            "HS",
            1,
            dilation,
        ),
        bneck_conf(
            dim1 * 2,
            5,
            dim3,
            dim1 * 2,
            True,
            "HS",
            1,
            dilation,
        ),
    ]
    last_channel = adjust_channels(dim3 * 2)  # C5

    return mobilenetv3._mobilenet_v3(
        inverted_residual_setting,
        last_channel,
        num_classes=num_classes,
        weights=None,
        progress=False,
        **kwargs,
    )


def _student_very_small_cifarv2(num_classes, dim1, dim2, dim3, **kwargs) -> nn.Module:
    dilation = 1

    width_mult = 1
    bneck_conf = partial(mobilenetv3.InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        mobilenetv3.InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    inverted_residual_setting = [
        # same conf as mobiletnet-s
        bneck_conf(16, 3, 16, 16, True, "RE", 1, 1),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 1, 1),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
        # modified conf
        bneck_conf(40, 5, 120, dim1, True, "HS", 1, 1),
        bneck_conf(dim1, 5, dim2, dim1, True, "HS", 1, 1),
        bneck_conf(dim1, 5, dim2 * 2, dim1 * 2, True, "HS", 2, dilation),  # C4
        bneck_conf(
            dim1 * 2,
            5,
            dim3,
            dim1 * 2,
            True,
            "HS",
            1,
            dilation,
        ),
        bneck_conf(
            dim1 * 2,
            5,
            dim3,
            dim1 * 2,
            True,
            "HS",
            1,
            dilation,
        ),
    ]
    last_channel = adjust_channels(dim3 * 2)  # C5

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

    MODEL_GENERATORS[f"student-mobilenetxs"] = partial(
        _student_very_small, dim1=24, dim2=144, dim3=128
    )

    MODEL_GENERATORS[f"student-mobilenetxxs"] = partial(
        _student_very_small, dim1=12, dim2=72, dim3=64
    )

    MODEL_GENERATORS[f"student-mobilenetxs-cifar"] = partial(
        _student_very_small_cifar, dim1=24, dim2=144, dim3=128
    )

    MODEL_GENERATORS[f"student-mobilenetxs-cifarv2"] = partial(
        _student_very_small_cifarv2, dim1=24, dim2=144, dim3=128
    )

    MODEL_GENERATORS[f"student-mobilenetxxs-cifar"] = partial(
        _student_very_small_cifar, dim1=12, dim2=72, dim3=64
    )
    MODEL_GENERATORS[f"student-mobilenetxxs-cifarv2"] = partial(
        _student_very_small_cifarv2, dim1=12, dim2=72, dim3=64
    )

    MODEL_GENERATORS[f"student-mobilenets-trained"] = _student_s_trained


_generate_model_function()
