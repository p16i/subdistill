import torch

import torchvision

from torch import nn

from . import resnet
from torchvision.models.resnet import ResNet18_Weights

MODEL_GENERATORS = dict()


def register_model(name):
    """Decorator to register a data modality provider."""

    def wrapped(fn):
        """Wrapped function to register a data modality provider with name `name`"""
        MODEL_GENERATORS[name] = fn

        return fn

    return wrapped


def get_model(name: str) -> nn.Module:
    dataset, arch, variant = name.split("-")

    # todo: better organizing these if-else structures
    if name in ["cifar10-resnet18-p1", "cifar100-resnet18-p1"]:
        assert variant == "p1", "We only have one variant for now!"

        num_classes = 10 if dataset == "cifar10" else 100

        model = MODEL_GENERATORS["cifar-resnet18"](num_classes=num_classes)

        if dataset == "cifar10":
            url = "https://tubcloud.tu-berlin.de/s/Ymy9WjzizxraqJy/download/resnet18-cifar10.pth"
        elif dataset == "cifar100":
            url = "https://tubcloud.tu-berlin.de/s/xZ29d76Sz29M9Qa/download/resnet18-cifar100.pth"
        else:
            raise ValueError(f"No checkpoint for `{name}`")

        model.load_state_dict(torch.hub.load_state_dict_from_url(url))

    elif name == "imagenet-resnet18-tv":
        model = MODEL_GENERATORS["imagenet-resnet18"]()
    else:
        raise ValueError(f"Unfortunately, we do NOT have a `{name}` model")

    setattr(model, "__name", name)
    setattr(model, "__layer_dimension", resnet.ARCH_LAYER_DIMENSIONS[arch])

    model.eval()

    return model


def get_layer_dimensions(model: nn.Module, layer: str) -> int:
    return getattr(model, "__layer_dimension")[layer]


@register_model("cifar-resnet18")
def _resnet18_cifar(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)

    # why we use this? (ask Florian?)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()

    model.avgpool = nn.AvgPool2d(kernel_size=4)
    model.fc = nn.Linear(512, num_classes)

    model.num_classes = num_classes

    return model


@register_model("imagenet-resnet18")
def _resnet18_cifar() -> nn.Module:
    return torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
