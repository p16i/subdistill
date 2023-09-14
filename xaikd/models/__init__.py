import typing
from typing import Callable, List, Optional, Type, Union
from collections import OrderedDict

import torch
import torch.nn as nn

import torchvision

from torch import nn
import numpy as np


from . import resnet, vgg, interfaces

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet18_Weights

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
    "cifar100-resnet18-wb5": "https://tubcloud.tu-berlin.de/s/CctcgcY489z86Nc/download/resnet18-cifar100-pretty-dust-5-adam.pth",
    "cifar100-resnet18-wb8": "https://tubcloud.tu-berlin.de/s/NfPMGt4fwA2mTxq/download/resnet18-cifar100-fragrant-frog-8.pth",
    "cifar100-resnet18-wb11": "https://tubcloud.tu-berlin.de/s/DpzZpcynKtWqgSj/download/resnet18-cifar100-devoted-music-11.pth",
    "cifar100-resnet18-wb15": "https://tubcloud.tu-berlin.de/s/Bm5AwmtGiYgD8Jx/download/cifar100-resnet18-whole-planet-15.pth",
    "cifar100-resnet18-wb15e197": "https://tubcloud.tu-berlin.de/s/aSz6NJnear5CNHE/download?path=%2F&files=model-lwnx8qeo:v24.pth",
    "cifar100-resnet18-wb15e151": "https://tubcloud.tu-berlin.de/s/aSz6NJnear5CNHE/download?path=%2F&files=model-lwnx8qeo:v19.pth",
    "cifar100-resnet18-wb15e121": "https://tubcloud.tu-berlin.de/s/aSz6NJnear5CNHE/download?path=%2F&files=model-lwnx8qeo%3Av10.pth",
    "cifar100-resnet18-wb15e61": "https://tubcloud.tu-berlin.de/s/aSz6NJnear5CNHE/download?path=%2F&files=model-lwnx8qeo:v9.pth",
    "cifar100-resnet18-wb15e23": "https://tubcloud.tu-berlin.de/s/aSz6NJnear5CNHE/download?path=%2F&files=model-lwnx8qeo:v6.pth",
    "cifar100-resnet18-wb15e1": "https://tubcloud.tu-berlin.de/s/aSz6NJnear5CNHE/download?path=%2F&files=model-lwnx8qeo:v0.pth",
    "cifar100-resnet18-wb22": "https://tubcloud.tu-berlin.de/s/wXy5HdfsT3CLRQN/download/cifar100-resnet18-wb22.pth",
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


def get_trained_model(name: str) -> interfaces.DistillableModel:
    dataset, arch, variant = name.split("-")

    # todo: better organizing these if-else structures
    if name in MODEL_CHECKPOINT_MAPPING.keys():
        num_classes = 10 if dataset == "cifar10" else 100

        model = MODEL_GENERATORS[f"cifar-{arch}"](num_classes=num_classes)

        url = MODEL_CHECKPOINT_MAPPING[name]

        model.load_state_dict(
            torch.hub.load_state_dict_from_url(url, file_name=f"{name}.pth")
        )

    elif name == "imagenet-resnet18-tv":
        model = MODEL_GENERATORS["imagenet-resnet18"]()
    elif name == "imagenet-vgg16-tv":
        model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
        model.num_classes = 1000
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
    if arch in ["vgg11", "vgg16"]:
        model = vgg.DistillableVGG.cast(model)
    elif arch in ["resnet18", "resnet50"]:
        model = resnet.DistillableResNet.cast(model)
    else:
        raise ValueError(f"`{model.__class__}` is NOT distillable!")

    model.num_classes = num_classes

    setattr(model, "__name", name)

    setattr(model, "__layer_dimension", constants.ARCH_LAYER_DIMENSIONS[arch])

    model.eval()

    assert getattr(model, "num_classes")

    # todo: disable grad
    # perhaps, check whether disable grad improve inference speed?

    return model


def get_untrained_model(name: str, num_classes: int):
    return MODEL_GENERATORS[name](num_classes=num_classes)


def get_layer_output_dimensions(model: nn.Module, layer: str) -> int:
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
    model = models.vgg.vgg11(num_classes=num_classes)

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
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    return model


def _generate_resnet18_compressed(
    compression_ratio: int, num_classes: int
) -> nn.Module:
    # todo: hard-corded everything for now.
    resnet18 = torchvision.models.resnet.resnet18()

    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L184
    ori_inplances = 64

    inplanes = int(ori_inplances / compression_ratio)
    # becuase inplance is modified throught the generation
    # we have to reset attribute
    resnet18.inplanes = inplanes

    # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
    layers = [
        # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
        ("conv1", nn.Conv2d(3, inplanes, 3, 1, 1, bias=False)),
        ("bn1", nn.BatchNorm2d(num_features=inplanes)),
        ("relu1", nn.ReLU()),
        ("maxpool", nn.Identity()),
    ]

    arr_num_blocks = [2, 2, 2, 2]
    arr_dims = inplanes * np.power(2, np.arange(4))

    for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
        layer = resnet18._make_layer(
            torchvision.models.resnet.BasicBlock,
            dims,
            num_blocks,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
            stride=2 if i > 0 else 1,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
            dilate=False,
        )

        # # todo: this is temporary;
        # todo: remove this after also in other derivative of the architecture.
        layers.append((f"layer{i+1}", layer))

    layers.extend(
        [
            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
        ]
    )

    model = nn.Sequential(OrderedDict(layers))

    return model


@register_model("resnet18compr2")
def _resnet18c2(num_classes: int):
    return _generate_resnet18_compressed(compression_ratio=2, num_classes=num_classes)


@register_model("resnet18compr4")
def _resnet18c4(num_classes: int):
    return _generate_resnet18_compressed(compression_ratio=4, num_classes=num_classes)


@register_model("resnet18compr8")
def _resnet18c8(num_classes: int):
    return _generate_resnet18_compressed(compression_ratio=8, num_classes=num_classes)


@register_model("resnet18customized")
def _generate_resnet18_customized(num_classes: int) -> nn.Module:
    # todo: hard-corded everything for now.
    resnet18 = torchvision.models.resnet.resnet18()

    inplanes = 32
    # becuase inplance is modified throught the generation
    # we have to reset attribute
    resnet18.inplanes = inplanes

    # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
    layers = [
        # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
        ("conv1", nn.Conv2d(3, inplanes, 3, 1, 1, bias=False)),
        ("bn1", nn.BatchNorm2d(num_features=inplanes)),
        ("relu1", nn.ReLU()),
        ("maxpool", nn.Identity()),
    ]

    arr_num_blocks = [2, 2, 2, 2]
    arr_dims = [32, 32, 48, 64]

    for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
        layer = resnet18._make_layer(
            torchvision.models.resnet.BasicBlock,
            dims,
            num_blocks,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
            stride=2 if i > 0 else 1,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
            dilate=False,
        )

        # todo: this is temporary
        layers.append(
            (f"layer{i+1}", nn.Sequential(layer, nn.BatchNorm2d(num_features=dims)))
        )

    layers.extend(
        [
            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
        ]
    )

    model = nn.Sequential(OrderedDict(layers))

    return model


@register_model("resnet18customized2")
def _generate_resnet18_customized2(num_classes: int) -> nn.Module:
    # todo: hard-corded everything for now.
    resnet18 = torchvision.models.resnet.resnet18()

    inplanes = 32
    # becuase inplance is modified throught the generation
    # we have to reset attribute
    resnet18.inplanes = inplanes

    # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
    layers = [
        # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
        ("conv1", nn.Conv2d(3, inplanes, 3, 1, 1, bias=False)),
        ("bn1", nn.BatchNorm2d(num_features=inplanes)),
        ("relu1", nn.ReLU()),
        ("maxpool", nn.Identity()),
    ]

    arr_num_blocks = [2, 2, 2, 2]
    arr_dims = [32, 32, 32, 32]

    for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
        layer = resnet18._make_layer(
            torchvision.models.resnet.BasicBlock,
            dims,
            num_blocks,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
            stride=2 if i > 0 else 1,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
            dilate=False,
        )

        # todo: this is temporary
        layers.append(
            (f"layer{i+1}", nn.Sequential(layer, nn.BatchNorm2d(num_features=dims)))
        )

    layers.extend(
        [
            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
        ]
    )

    model = nn.Sequential(OrderedDict(layers))

    return model


def _generate_resnet18_imagenet_compressed(
    compression_ratio: int, num_classes: int
) -> nn.Module:
    # todo: hard-corded everything for now.
    resnet18 = torchvision.models.resnet.resnet18()

    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L184
    ori_inplances = 64

    inplanes = int(ori_inplances / compression_ratio)
    # becuase inplance is modified throught the generation
    # we have to reset attribute
    resnet18.inplanes = inplanes

    layers = [
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L197
        (
            "conv1",
            nn.Conv2d(3, inplanes, kernel_size=7, padding=3, stride=2, bias=False),
        ),
        ("bn1", nn.BatchNorm2d(num_features=inplanes)),
        ("relu1", nn.ReLU()),
        ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]

    arr_num_blocks = [2, 2, 2, 2]
    arr_dims = inplanes * np.power(2, np.arange(4))

    for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
        layer = resnet18._make_layer(
            torchvision.models.resnet.BasicBlock,
            dims,
            num_blocks,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
            stride=2 if i > 0 else 1,
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
            dilate=False,
        )

        # todo: this is temporary
        layers.append(
            (f"layer{i+1}", nn.Sequential(layer, nn.BatchNorm2d(num_features=dims)))
        )

    layers.extend(
        [
            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
        ]
    )

    model = nn.Sequential(OrderedDict(layers))

    return model


@register_model("resnet18imagenetcompr2")
def _resnet18imagenetc2(num_classes: int):
    return _generate_resnet18_imagenet_compressed(
        compression_ratio=2, num_classes=num_classes
    )
