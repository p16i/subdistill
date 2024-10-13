import numpy as np

import typing


from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    MobileNetV3,
)

import torch

from torch import nn


from . import interfaces
from . import register_model


def split_model_at(
    model: MobileNetV3, layer: str
) -> typing.Tuple[nn.Sequential, nn.Sequential]:
    assert isinstance(model, MobileNetV3)

    assert len(layer.split(".")) == 2

    layer_ix = int(layer.split(".")[-1])

    assert layer_ix >= 0

    layers = model.features

    # remark: we use zero-index; the head part therefore also includes with features.ix
    layers_in_head = layers[: layer_ix + 1]
    layers_in_classifier = layers[layer_ix + 1 :]

    head = nn.Sequential(*layers_in_head)

    classifier = nn.Sequential(
        *layers_in_classifier, model.avgpool, nn.Flatten(start_dim=1), model.classifier
    )

    return head, classifier


@register_model("imagenet-mobilenetl-tv")
def _mobilenetl() -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    setattr(model, "__last_layer", model.classifier[3])

    return model


@register_model("imagenet-mobilenets-tv")
def _mobilenets() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    setattr(model, "__last_layer", model.classifier[3])

    return model
