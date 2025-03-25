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

# These paths are from https://tubcloud.tu-berlin.de/apps/files/files/3567344323?dir=/projects/2023-knowledge-distillation
# Remark: The URLs are different because they have been generated at different times and different roots.
MODEL_CHECKPOINT_MAPPING = {
    "cifar100-resnet18-v1": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-resnet18-v1--model-sszu9jtz:best.pth",
    "cifar100-resnet18-v2": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-resnet18-v2--model-8no232l1:best.pth",
    "cifar100-resnet50-v1": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-resnet50-v1--model-dxngvotm:best.pth",
    "cifar100-vgg11-v1": "https://tubcloud.tu-berlin.de/s/YXQWsGmz4kRnfLL/download?path=%2F&files=cifar100-vgg11-v1--model-rm0pe4r0:best.pth",
    "celeba-resnet18-scratch": "https://tubcloud.tu-berlin.de/s/Ej2KoCpTtpZ3g6r/download?path=%2Fceleba&files=celeba--scratch--n8r0q2vb.pth",
    "celeba-resnet18-pretrained": "https://tubcloud.tu-berlin.de/s/Ej2KoCpTtpZ3g6r/download?path=%2Fceleba&files=celeba--imagenet-pretrained--6oj5aaxl.pth",
    "celeba-resnet18-finetunedv1": "https://tubcloud.tu-berlin.de/s/Ej2KoCpTtpZ3g6r/download?path=%2Fceleba%2F/finetuned-from-imagenet&files=resnet18--p16i-xaikd-training-teacher-models-zbgow8eu.pth",
    "celeba-resnet50-finetunedv1": "https://tubcloud.tu-berlin.de/s/Ej2KoCpTtpZ3g6r/download?path=%2Fceleba%2F/finetuned-from-imagenet&files=resnet50--p16i-xaikd-training-teacher-models-cuoynabf.pth",
    "celeba-wideresnet50_2-finetunedv1": "https://tubcloud.tu-berlin.de/s/Ej2KoCpTtpZ3g6r/download?path=%2Fceleba%2F/finetuned-from-imagenet&files=wideresnet50-2--p16i-xaikd-training-teacher-models-t0sg5wcp.pth",
    "celeba-vitb16-finetunedv1": "https://tubcloud.tu-berlin.de/s/Ej2KoCpTtpZ3g6r/download?path=%2Fceleba%2F/finetuned-from-imagenet&files=vitb16--p16i-xaikd-training-teacher-models-6ttr2icx.pth",
}


def add_model_to_registry(name: str, fn: Callable):
    assert name not in MODEL_GENERATORS

    MODEL_GENERATORS[name] = fn


def register_model(name):
    """Decorator to register a data modality provider."""

    def wrapped(fn):
        """Wrapped function to register a data modality provider with name `name`"""

        add_model_to_registry(name, fn)

        return fn

    return wrapped


def get_trained_model(name: str) -> nn.Module:
    dataset, arch, variant = name.split("-")

    # todo: better organizing these if-else structures
    if name in MODEL_CHECKPOINT_MAPPING.keys():
        url = MODEL_CHECKPOINT_MAPPING[name]

        if "cifar" in dataset:
            num_classes = 10 if dataset == "cifar10" else 100

            model = MODEL_GENERATORS[f"cifar-{arch}"](num_classes=num_classes)
        elif dataset == "celeba":
            CELEBA_NUM_ATTRIBUTES = 40
            if arch == "resnet18":
                model = torchvision.models.resnet18(
                    weights=None, num_classes=CELEBA_NUM_ATTRIBUTES
                )
            elif arch == "resnet50":
                model = torchvision.models.resnet50(
                    weights=None, num_classes=CELEBA_NUM_ATTRIBUTES
                )
            elif arch == "wideresnet50_2":
                model = torchvision.models.wide_resnet50_2(
                    weights=None, num_classes=CELEBA_NUM_ATTRIBUTES
                )
            elif arch == "vitb16":
                model = torchvision.models.vit_b_16(
                    weights=None, num_classes=CELEBA_NUM_ATTRIBUTES
                )
            else:
                raise ValueError(f"`arhc={name}` doesn't exist")
            setattr(model, "num_classes", CELEBA_NUM_ATTRIBUTES)
        else:
            raise ValueError(f"`{name}` doesn't exist")

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

    model.eval()

    assert getattr(model, "num_classes")

    return model


def get_untrained_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    print(f"Constructing untrain-model={name} with ( {num_classes} outputs)")
    return MODEL_GENERATORS[name](num_classes=num_classes, **kwargs)


def get_layer_output_dimensions(model: nn.Module, layer: str) -> int:
    raise
    return getattr(model, "__layer_dimension")[layer]


from . import resnet, vgg, nfnet, vit, mobilenets, students, layers


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
