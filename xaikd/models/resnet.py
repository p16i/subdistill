import numpy as np

import typing

import types
from functools import partial

from collections import OrderedDict

import torch

from torch import nn

import torchvision
from torchvision.models import resnet

from xaikd.datasets.celeba import NUM_CELEBA_ATTRIBUTES

from . import interfaces
from . import register_model, add_model_to_registry, MODEL_CHECKPOINT_MAPPING


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


def _construct_resnet18_cifar(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)

    # Refs:
    # - SimCLR, Appendix B.9 CIFAR10
    # - https://github.com/p3i0t/SimCLR-CIFAR10/blob/master/models.py#L12
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()  # type: ignore

    model.num_classes = num_classes  # type: ignore

    return model


@register_model("cifar100-resnet18-v1")
def _resnet18_cifar100_v1() -> nn.Module:
    name = "cifar100-resnet18-v1"
    model = _construct_resnet18_cifar(num_classes=100)

    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_CHECKPOINT_MAPPING[name], file_name=f"{name}.pth"
        )
    )

    return model


@register_model("cifar100-resnet18-v2")
def _resnet18_cifar100_v2() -> nn.Module:
    name = "cifar100-resnet18-v2"
    model = _construct_resnet18_cifar(num_classes=100)

    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_CHECKPOINT_MAPPING[name], file_name=f"{name}.pth"
        )
    )

    return model


@register_model("imagenet-resnet18-tv")
def _resnet18_imagenet() -> nn.Module:
    model = torchvision.models.resnet18(weights=resnet.ResNet18_Weights.IMAGENET1K_V1)
    model.num_classes = 1000  # type: ignore

    return model


@register_model("imagenet-resnet34-tv")
def _resnet34_imagenet() -> nn.Module:
    model = torchvision.models.resnet34(weights=resnet.ResNet34_Weights.IMAGENET1K_V1)
    model.num_classes = 1000  # type: ignore

    return model


@register_model("imagenet-resnet50-tv")
def _resnet50_imagenet() -> nn.Module:
    model = torchvision.models.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
    model.num_classes = 1000  # type: ignore

    return model


@register_model("imagenet-resnet101-tv")
def _resnet101_imagenet() -> nn.Module:
    model = torchvision.models.resnet101(weights=resnet.ResNet101_Weights.IMAGENET1K_V1)
    model.num_classes = 1000  # type: ignore

    return model


@register_model("imagenet-resnet152-tv")
def _resnet152_imagenet() -> nn.Module:
    model = torchvision.models.resnet152(weights=resnet.ResNet152_Weights.IMAGENET1K_V1)
    model.num_classes = 1000  # type: ignore

    return model


@register_model("celeba-resnet18-finetunedv1")
def _resnet18_celeba() -> nn.Module:

    name = "celeba-resnet18-finetunedv1"
    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CELEBA_ATTRIBUTES)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_CHECKPOINT_MAPPING[name], file_name=f"{name}.pth"
        )
    )
    model.num_classes = NUM_CELEBA_ATTRIBUTES  # type: ignore
    return model


@register_model("celeba-resnet50-finetunedv1")
def _resnet50_celeba() -> nn.Module:

    name = "celeba-resnet50-finetunedv1"
    model = torchvision.models.resnet50(weights=None, num_classes=NUM_CELEBA_ATTRIBUTES)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_CHECKPOINT_MAPPING[name], file_name=f"{name}.pth"
        )
    )
    model.num_classes = NUM_CELEBA_ATTRIBUTES  # type: ignore
    return model


@register_model("celeba-wideresnet50_2-finetunedv1")
def _wideresnet50_2_celeba() -> nn.Module:

    name = "celeba-wideresnet50_2-finetunedv1"
    model = torchvision.models.wide_resnet50_2(
        weights=None, num_classes=NUM_CELEBA_ATTRIBUTES
    )
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            MODEL_CHECKPOINT_MAPPING[name], file_name=f"{name}.pth"
        )
    )
    model.num_classes = NUM_CELEBA_ATTRIBUTES  # type: ignore
    return model


def construct_student_cifar_resnet18(in_planes: int, num_classes: int, **kwargs):
    model = resnet._resnet(
        resnet.BasicBlock,
        [2, 2, 2, 2],
        weights=None,
        progress=False,
        num_classes=num_classes,
    )

    model.inplanes = in_planes

    # Similar to _resnet18_cifar(..)
    model.conv1 = nn.Conv2d(3, in_planes, 3, 1, 1, bias=False)
    model.bn1 = nn.BatchNorm2d(in_planes)
    model.maxpool = nn.Identity()  # type: ignore

    # The following code mimics the original code's _make_layer(..)
    # Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L225

    # We only change in planes
    model.inplanes = in_planes
    model.layer1 = model._make_layer(
        block=resnet.BasicBlock,
        planes=in_planes,
        blocks=2,
    )

    model.layer2 = model._make_layer(
        block=resnet.BasicBlock, planes=in_planes, blocks=2, stride=2, dilate=False
    )

    model.inplanes = in_planes
    model.layer3 = model._make_layer(
        block=resnet.BasicBlock, planes=in_planes, blocks=2, stride=2, dilate=False
    )

    model.inplanes = in_planes
    model.layer4 = model._make_layer(
        block=resnet.BasicBlock, planes=in_planes, blocks=2, stride=2, dilate=False
    )

    model.fc = nn.Linear(model.inplanes, num_classes)

    return model


def construct_student_resnet18(in_planes: int, num_classes: int, **kwargs):
    model = resnet._resnet(
        resnet.BasicBlock,
        [2, 2, 2, 2],
        weights=None,
        progress=False,
        num_classes=num_classes,
    )

    model.inplanes = in_planes
    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L196
    model.conv1 = nn.Conv2d(
        3, in_planes, kernel_size=7, stride=2, padding=3, bias=False
    )
    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L198
    model.bn1 = nn.BatchNorm2d(in_planes)

    # The following code mimics the original code's _make_layer(..)
    # Ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L225
    # Similar to _resnet18_cifar(..)

    # We only change in planes
    model.inplanes = in_planes
    model.layer1 = model._make_layer(
        block=resnet.BasicBlock,
        planes=in_planes,
        blocks=2,
    )

    model.layer2 = model._make_layer(
        block=resnet.BasicBlock, planes=in_planes, blocks=2, stride=2, dilate=False
    )

    model.inplanes = in_planes
    model.layer3 = model._make_layer(
        block=resnet.BasicBlock, planes=in_planes, blocks=2, stride=2, dilate=False
    )

    model.inplanes = in_planes
    model.layer4 = model._make_layer(
        block=resnet.BasicBlock, planes=in_planes, blocks=2, stride=2, dilate=False
    )

    model.fc = nn.Linear(model.inplanes, num_classes)

    return model


def _register_student_resnet18():

    for in_planes in [16, 32, 64]:

        add_model_to_registry(
            f"student-cifar-resnet18-{in_planes}",
            partial(construct_student_cifar_resnet18, in_planes=in_planes),
        )
        add_model_to_registry(
            f"student-resnet18-{in_planes}",
            partial(construct_student_resnet18, in_planes=in_planes),
        )


_register_student_resnet18()
