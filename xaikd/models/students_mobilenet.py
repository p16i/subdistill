import typing
import torch
from torchvision.models import (
    mobilenetv3,
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)
from torchvision.ops import Conv2dNormActivation

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


def _student_very_small_bottleneck(
    num_classes, dim1, dim2, dim3, dim4, last_output_channels, **kwargs
) -> nn.Module:
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
    last_channel = adjust_channels(dim4)  # C5

    # return mobilenetv3._mobilenet_v3(
    return PatMobileNetV3(
        inverted_residual_setting,
        last_conv_channels=last_output_channels,
        last_channel=last_channel,
        num_classes=num_classes,
        weights=None,
        progress=False,
        **kwargs,
    )


class PatMobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: typing.List[mobilenetv3.InvertedResidualConfig],
        last_conv_channels: int,
        last_channel: int,
        num_classes: int = 1000,
        block=None,
        norm_layer=None,
        dropout=0.2,
        **kwargs,
    ) -> None:

        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, typing.Sequence)
            and all(
                [
                    isinstance(s, mobilenetv3.InvertedResidualConfig)
                    for s in inverted_residual_setting
                ]
            )
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]"
            )

        if block is None:
            block = mobilenetv3.InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_conv_channels
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


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

    # fixme: use register function
    MODEL_GENERATORS[f"student-mobilenets"] = _student_s
    MODEL_GENERATORS[f"student-mobilenets-lastd25"] = partial(
        _student_very_small_bottleneck,
        dim1=48,
        dim2=30,
        dim3=40,
        last_output_channels=25,
        dim4=16,
    )

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
