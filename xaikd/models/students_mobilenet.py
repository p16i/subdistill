from torch import nn
import timm

from functools import partial
import math
from . import add_model_to_registry


def __mobilenetv4_small_timm_alternative_initialization(**kwargs):
    model = timm.create_model("mobilenetv4_conv_small", **kwargs)

    # remark: it seems that the initiliazation from timm doesn't seem to work well with small data.
    # From Pat's preliminary invesgiation, the problem might be related to
    # the correction of the `fan_out` value when groups > 1 [1].
    # To the issue, we use the initalization scheme in [2].
    # Refs:
    #   [1] https://github.com/rwightman/timm/blob/main/timm/models/_efficientnet_builder.py#L551
    #   [2] https://github.com/d-li14/mobilenetv4.pytorch/blob/main/mobilenetv4.py#L155C19-L155C47
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    return model


def __mobilenetv4_small(**kwargs):
    model = timm.create_model("mobilenetv4_conv_small", **kwargs)
    return model


def get_mobilenetv4_small_with_factor(factor, **kwargs):
    model = timm.models.mobilenetv3._gen_mobilenet_v4(
        "mobilenetv4_conv_small", factor, pretrained=False, **kwargs
    )

    return model


def _generate_model_function():
    add_model_to_registry(
        "student-mobilenetv4-small",
        __mobilenetv4_small,
    )

    for factor in [0.125, 0.25, 0.5]:
        add_model_to_registry(
            f"student-mobilenetv4-small-{factor}",
            partial(get_mobilenetv4_small_with_factor, factor=factor),
        )

    add_model_to_registry(
        "student-mobilenetv4-small-alternative-init",
        __mobilenetv4_small_timm_alternative_initialization,
    )


_generate_model_function()
