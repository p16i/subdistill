import torch

from collections import OrderedDict

import typing
from torch import nn


from torchvision import models

from . import register_model, interfaces


def split_model_at(
    model: models.VGG, layer: str
) -> typing.Tuple[nn.Sequential, nn.Sequential]:
    assert isinstance(model, models.VGG)

    # layer in format `features.k``
    layer_ix = int(layer.split(".")[1])

    layers_in_head = model.features[: layer_ix + 1]
    layers_in_classifier = model.features[layer_ix + 1 :]

    head = nn.Sequential(*layers_in_head)

    classifier = nn.Sequential(
        *layers_in_classifier,
        model.avgpool,
        nn.Flatten(start_dim=1),
        model.classifier,
    )

    return head, classifier


@register_model("imagenet-vgg11-tv")
def _vgg11_imagenet() -> nn.Module:
    model = models.vgg11(weights=models.vgg.VGG11_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg11bn-tv")
def _vgg11bn_imagenet() -> nn.Module:
    model = models.vgg11_bn(weights=models.vgg.VGG11_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg13-tv")
def _vgg13_imagenet() -> nn.Module:
    model = models.vgg13(weights=models.vgg.VGG13_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg13bn-tv")
def _vgg13bn_imagenet() -> nn.Module:
    model = models.vgg13_bn(weights=models.vgg.VGG13_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg16-tv")
def _vgg16_imagenet() -> nn.Module:
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg16bn-tv")
def _vgg16bn_imagenet() -> nn.Module:
    model = models.vgg16_bn(weights=models.vgg.VGG16_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg19-tv")
def _vgg19_imagenet() -> nn.Module:
    model = models.vgg19(weights=models.vgg.VGG19_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg19bn-tv")
def _vgg19bn_imagenet() -> nn.Module:
    model = models.vgg19_bn(weights=models.vgg.VGG19_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model
