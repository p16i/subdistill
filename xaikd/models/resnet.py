import numpy as np

import typing

import types

from collections import OrderedDict

import torch

from torch import nn

import torchvision
from torchvision.models import resnet

from . import interfaces
from . import register_model


def split_model_at(
    model: resnet.ResNet, layer: str
) -> typing.Tuple[nn.Sequential, nn.Sequential]:
    assert isinstance(model, resnet.ResNet)

    assert len(layer.split(".")) == 1

    layer_ix = int(layer[-1])

    layers = [model.layer1, model.layer2, model.layer3, model.layer4]

    layers_in_head = layers[:layer_ix] if layer_ix > 0 else []
    layers_in_classifier = layers[layer_ix:]

    head = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool, *layers_in_head
    )

    classifier = nn.Sequential(
        *layers_in_classifier, model.avgpool, nn.Flatten(start_dim=1), model.fc
    )

    return head, classifier


@register_model("cifar-resnet18")
def _resnet18_cifar(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)

    # why we use this? (ask Florian?)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()

    model.avgpool = nn.AvgPool2d(kernel_size=4)

    model.num_classes = num_classes

    return model


@register_model("cifar-resnet50")
def _resnet50_cifar(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet50(weights=None, num_classes=num_classes)

    # why we use this? (ask Florian?)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()

    model.avgpool = nn.AvgPool2d(kernel_size=4)

    model.num_classes = num_classes

    return model


@register_model("imagenet-resnet18-tv")
def _resnet18_imagenet() -> nn.Module:
    model = torchvision.models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    return model


@register_model("imagenet-resnet34-tv")
def _resnet34_imagenet() -> nn.Module:
    model = torchvision.models.resnet34(weights=resnet.ResNet34_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    return model


@register_model("imagenet-resnet50-tv")
def _resnet50_imagenet() -> nn.Module:
    model = torchvision.models.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    return model


@register_model("imagenet-resnet101-tv")
def _resnet101_imagenet() -> nn.Module:
    model = torchvision.models.resnet101(weights=resnet.ResNet101_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    return model


@register_model("imagenet-resnet152-tv")
def _resnet152_imagenet() -> nn.Module:
    model = torchvision.models.resnet152(weights=resnet.ResNet152_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    return model
