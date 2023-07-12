import torch

import torchvision

from torch import nn

from . import resnet

from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import vgg
from torchvision import models

from xaikd import constants

MODEL_GENERATORS = dict()

MODEL_CHECKPOINT_MAPPING = {
    "cifar10-resnet18-p1": "https://tubcloud.tu-berlin.de/s/Ymy9WjzizxraqJy/download/resnet18-cifar10.pth",
    "cifar10-resnet18-simclr_finetuned_all1": "https://tubcloud.tu-berlin.de/s/qBY4b3tA3L9ZYN9/download/resnet18simclr-finetune-all-cifar10-seed1.pth",
    "cifar10-resnet18-simclr_finetuned_fc1": "https://tubcloud.tu-berlin.de/s/bwrNFY36KSy7KYj/download/resnet18simclr-finetune-fc-cifar10-seed1.pth",
    "cifar100-resnet18-p1": "https://tubcloud.tu-berlin.de/s/xZ29d76Sz29M9Qa/download/resnet18-cifar100.pth",
    "cifar100-resnet18-p2": "https://tubcloud.tu-berlin.de/s/82DSTLJppJfGesc/download/resnet18-cifar100-seed2.pth",
    "cifar100-resnet18-p3": "https://tubcloud.tu-berlin.de/s/E2KLikTmZCsbEqK/download/resnet18-cifar100-seed3.pth",
    "cifar100-resnet50-p1": "https://tubcloud.tu-berlin.de/s/FCefnjtD3KyRFRs/download/resnet50-cifar100-seed1.pth",
    "cifar100-vgg11-p1": "https://tubcloud.tu-berlin.de/s/xDbi6DsjyPppi3B/download/vgg11-cifar100-seed1.pth",
}


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
    if name in MODEL_CHECKPOINT_MAPPING.keys():
        num_classes = 10 if dataset == "cifar10" else 100

        model = MODEL_GENERATORS[f"cifar-{arch}"](num_classes=num_classes)

        url = MODEL_CHECKPOINT_MAPPING[name]

        model.load_state_dict(torch.hub.load_state_dict_from_url(url))

    elif name == "imagenet-resnet18-tv":
        model = MODEL_GENERATORS["imagenet-resnet18"]()
    elif "imagenet-resnet18-random" in name:
        # use regex to parse the number
        seed = int(name.split("-")[-1].replace("random", ""))
        print(f"Using Random `resnet18(seed={seed})` Model")
        torch.manual_seed(seed)
        model = torchvision.models.resnet18()
    elif name == "imagenet-vgg16-tv":
        model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    elif "imagenet-vgg16-random" in name:
        seed = int(name.split("-")[-1].replace("random", ""))
        print(f"Using Random `vgg16(seed={seed})` Model")
        torch.manual_seed(seed)
        model = models.vgg16()
    else:
        raise ValueError(f"Unfortunately, we do NOT have a `{name}` model")

    setattr(model, "__name", name)

    setattr(model, "__layer_dimension", constants.ARCH_LAYER_DIMENSIONS[arch])

    model.eval()

    # todo: disable grad
    # perhaps, check whether disable grad improve inference speed?

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


@register_model("cifar-vgg11")
def _cifar_vgg11(num_classes: int) -> nn.Module:
    model = vgg.vgg11()
    model.classifier[6] = nn.Linear(4096, num_classes)

    model.num_classes = num_classes

    return model


@register_model("cifar-resnet50")
def _resnet50_cifar(num_classes: int) -> nn.Module:
    model = torchvision.models.resnet50(weights=None)

    # why we use this? (ask Florian?)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()

    model.avgpool = nn.AvgPool2d(kernel_size=4)
    model.fc = nn.Linear(2048, num_classes)

    model.num_classes = num_classes

    return model


@register_model("imagenet-resnet18")
def _resnet18_imagenet() -> nn.Module:
    return torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
