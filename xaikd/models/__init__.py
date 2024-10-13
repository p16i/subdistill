import typing
from typing import Callable, List, Optional, Type, Union
from collections import OrderedDict

import torch
import torch.nn as nn

import torchvision

from torch import nn
import numpy as np


from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet18_Weights

from torchvision import models

from xaikd import constants

MODEL_GENERATORS = dict()

MODEL_CHECKPOINT_MAPPING = {
    "cifar100-resnet18-v1": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-resnet18-v1--model-sszu9jtz:best.pth",
    "cifar100-resnet18-v2": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-resnet18-v2--model-8no232l1:best.pth",
    "cifar100-resnet50-v1": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-resnet50-v1--model-dxngvotm:best.pth",
    "cifar100-vgg11-v1": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-vgg11-v1--model-rm0pe4r0:best.pth",
}


def register_model(name):
    """Decorator to register a data modality provider."""

    def wrapped(fn):
        """Wrapped function to register a data modality provider with name `name`"""
        MODEL_GENERATORS[name] = fn

        return fn

    return wrapped


def get_trained_model(name: str) -> nn.Module:
    dataset, arch, variant = name.split("-")

    # todo: better organizing these if-else structures
    if name in MODEL_CHECKPOINT_MAPPING.keys():
        num_classes = 10 if dataset == "cifar10" else 100

        model = MODEL_GENERATORS[f"cifar-{arch}"](num_classes=num_classes)

        url = MODEL_CHECKPOINT_MAPPING[name]

        model.load_state_dict(
            torch.hub.load_state_dict_from_url(url, file_name=f"{name}.pth")
        )
    elif name in MODEL_GENERATORS.keys():
        model = MODEL_GENERATORS[name]()
    elif "imagenet-resnet18-random" in name:
        # use regex to parse the number
        seed = int(name.split("-")[-1].replace("random", ""))
        print(f"Using Random `resnet18(seed={seed})` Model")
        torch.manual_seed(seed)
        model = torchvision.models.resnet18()
    elif "imagenet-vgg16-random" in name:
        seed = int(name.split("-")[-1].replace("random", ""))
        print(f"Using Random `vgg16(seed={seed})` Model")
        torch.manual_seed(seed)
        model = models.vgg16()
    else:
        raise ValueError(f"Unfortunately, we do NOT have a `{name}` model")

    num_classes = model.num_classes

    # cast native torchvision model to our `DistillableModel`
    if "vgg" in arch:
        setattr(model, "__last_layer", model.classifier[-1])
    elif "resnet" in arch:
        setattr(model, "__last_layer", model.fc)

    model.num_classes = num_classes

    setattr(model, "__name", name)

    setattr(model, "__layer_dimension", constants.ARCH_LAYER_DIMENSIONS[arch])

    model.eval()

    assert getattr(model, "num_classes")

    # todo: disable grad
    # perhaps, check whether disable grad improve inference speed?

    return model


def get_untrained_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    return MODEL_GENERATORS[name](num_classes=num_classes, **kwargs)


def get_layer_output_dimensions(model: nn.Module, layer: str) -> int:
    raise
    return getattr(model, "__layer_dimension")[layer]


from . import resnet, vgg, nfnet, vit, mobilenets, students


def split_model_at_layer(model, layer: str) -> typing.Tuple[nn.Module, nn.Module]:
    if isinstance(model, resnet.resnet.ResNet):
        return resnet.split_model_at(model, layer)
    elif isinstance(model, vgg.models.VGG):
        return vgg.split_model_at(model, layer)
    elif isinstance(model, nfnet.NormFreeNet):
        return nfnet.split_model_at(model, layer)
    elif isinstance(model, vit.VisionTransformer):
        return vit.split_model_at(model, layer)
    else:
        raise ValueError(f"no available split_model for layer={layer} model={model}")
