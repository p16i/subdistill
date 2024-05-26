import torch

from collections import OrderedDict

import typing
from torch import nn


from torchvision import models

from . import register_model, interfaces


from xaikd.utils.modules import (
    Centering2D,
    DiagonalScaling,
    merge_conv_and_bn,
    merge_convKxK_and_conv1x1,
)


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
