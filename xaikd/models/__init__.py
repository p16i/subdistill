import torch

import torchvision

from torch import nn

from . import resnet


def get_model(slug: str):
    # should we return transformations?
    # todo: add return type
    # todo: better organizing these if-else structures
    if slug in ["cifar10-resnet18-p1", "cifar100-resnet18-p1"]:
        dataset, arch, variant = slug.split("-")

        num_classes = 10 if dataset == "cifar10" else 100

        model = torchvision.models.resnet18(weights=None)

        # why we use this? (ask Florian?)
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()

        model.avgpool = nn.AvgPool2d(kernel_size=4)
        model.fc = nn.Linear(512, num_classes)

        model.num_classes = num_classes

        assert variant == "p1", "We only have one variant for now!"

        if dataset == "cifar10":
            url = "https://tubcloud.tu-berlin.de/s/Ymy9WjzizxraqJy/download/resnet18-cifar10.pth"
        elif dataset == "cifar100":
            url = "https://tubcloud.tu-berlin.de/s/xZ29d76Sz29M9Qa/download/resnet18-cifar100.pth"
        else:
            raise ValueError(f"No checkpoint for `{slug}`")

        model.load_state_dict(torch.hub.load_state_dict_from_url(url))

    else:
        raise ValueError(f"Unfortunately, we do NOT have a `{slug}` model")

    model.eval()

    return model


def get_layer_dimensions(arch: str, layer: str) -> int:
    if arch == "resnet18":
        return resnet.ARCH_LAYER_DIMENSIONS["resnet18"][layer]
    else:
        raise NotImplementedError()
