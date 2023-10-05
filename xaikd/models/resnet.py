import numpy as np

import typing

import types

from collections import OrderedDict

from torchvision.models.resnet import ResNet18_Weights
import torch

from torch import nn

import torchvision
from torchvision.models import resnet

from . import interfaces
from . import register_model


class DiagonalScaling(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()

        self.scale = nn.Parameter(torch.randn(dims).reshape(1, dims, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dims).reshape(1, dims, 1, 1))

    def forward(self, x):
        return self.scale * x + self.bias


class DistillableResNet(interfaces.DistillableModel, resnet.ResNet):
    @classmethod
    def cast(cls, model: resnet.ResNet):
        assert isinstance(model, resnet.ResNet)

        model.__class__ = cls

        assert isinstance(model, DistillableResNet)

        return model

    def split_at(self, layer: str) -> typing.Tuple[nn.Module, nn.Module, nn.Module]:
        assert len(layer.split(".")) == 1

        assert isinstance(self, resnet.ResNet)

        layer_ix = int(layer[-1]) - 1

        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        layers_in_head = layers[:layer_ix] if layer_ix > 0 else []
        layers_in_classifier = layers[layer_ix + 1 :]

        head = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, *layers_in_head
        )

        layer_module = layers[layer_ix]

        classifier = nn.Sequential(
            *layers_in_classifier, self.avgpool, nn.Flatten(start_dim=1), self.fc
        )

        return head, layer_module, classifier


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


@register_model("imagenet-resnet18")
def _resnet18_imagenet() -> nn.Module:
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    return model


def _generate_resnet18_compressed(
    compression_ratio: int, num_classes: int, for_cifar: bool
) -> nn.Module:
    # todo: hard-corded everything for now.
    resnet18 = torchvision.models.resnet.resnet18()

    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L184

    inplanes = 32 // compression_ratio
    # becuase inplance is modified throught the generation
    # we have to reset attribute
    resnet18.inplanes = inplanes

    layers = []

    if for_cifar:
        # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
        layers.extend(
            [
                # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
                (
                    "conv1",
                    nn.Conv2d(
                        3, inplanes, kernel_size=3, padding=1, stride=1, bias=False
                    ),
                ),
                ("bn1", nn.BatchNorm2d(num_features=inplanes)),
                ("relu1", nn.ReLU()),
                ("maxpool", nn.Identity()),
            ]
        )
    else:
        layers.extend(
            [
                # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L197
                (
                    "conv1",
                    nn.Conv2d(
                        3, inplanes, kernel_size=7, padding=3, stride=2, bias=False
                    ),
                ),
                ("bn1", nn.BatchNorm2d(num_features=inplanes)),
                ("relu1", nn.ReLU()),
                ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]
        )

    arr_num_blocks = [2, 2, 2, 2]
    arr_dims = [
        32 // compression_ratio,
        64 // compression_ratio,
        64 // compression_ratio,
        64 // compression_ratio,
    ]

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
        layers.append(
            (
                f"layer{i+1}",
                nn.Sequential(layer, DiagonalScaling(dims=dims)),
            )
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


@register_model("resnet18xscifarcompr1")
def _cifarresnet18c1(num_classes: int):
    return _generate_resnet18_compressed(
        compression_ratio=1, num_classes=num_classes, for_cifar=True
    )


@register_model("resnet18xscifarcompr2")
def _cifarresnet18c2(num_classes: int):
    return _generate_resnet18_compressed(
        compression_ratio=2, num_classes=num_classes, for_cifar=True
    )


@register_model("resnet18xscifarcompr4")
def _cifarresnet18c4(num_classes: int):
    return _generate_resnet18_compressed(
        compression_ratio=4, num_classes=num_classes, for_cifar=True
    )


# @register_model("resnet18cifarcompr8")
# def _cifarresnet18c8(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=8, num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18cifarcompr16")
# def _cifarresnet18c16(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=16, num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18cifarcompr32")
# def _cifarresnet18c32(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=32, num_classes=num_classes, for_cifar=True
#     )


# # @register_model("resnet18cifarcustomized")
# # def _generate_resnet18_customized(num_classes: int) -> nn.Module:
# #     # todo: hard-corded everything for now.
# #     resnet18 = torchvision.models.resnet.resnet18()

# #     inplanes = 32
# #     # becuase inplance is modified throught the generation
# #     # we have to reset attribute
# #     resnet18.inplanes = inplanes

# #     # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
# #     layers = [
# #         # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
# #         ("conv1", nn.Conv2d(3, inplanes, 3, 1, 1, bias=False)),
# #         ("bn1", nn.BatchNorm2d(num_features=inplanes)),
# #         ("relu1", nn.ReLU()),
# #         ("maxpool", nn.Identity()),
# #     ]

# #     arr_num_blocks = [2, 2, 2, 2]
# #     arr_dims = [32, 32, 48, 64]

# #     for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
# #         layer = resnet18._make_layer(
# #             torchvision.models.resnet.BasicBlock,
# #             dims,
# #             num_blocks,
# #             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
# #             stride=2 if i > 0 else 1,
# #             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
# #             dilate=False,
# #         )

# #         layers.append((f"layer{i+1}", layer))

# #     layers.extend(
# #         [
# #             ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
# #             ("flatten", nn.Flatten(start_dim=1)),
# #             ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
# #         ]
# #     )

# #     model = nn.Sequential(OrderedDict(layers))

# #     return model


# # @register_model("resnet18cifarcustomized2")
# # def _generate_resnet18_customized2(num_classes: int) -> nn.Module:
# #     # todo: hard-corded everything for now.
# #     resnet18 = torchvision.models.resnet.resnet18()

# #     inplanes = 32
# #     # becuase inplance is modified throught the generation
# #     # we have to reset attribute
# #     resnet18.inplanes = inplanes

# #     # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
# #     layers = [
# #         # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
# #         ("conv1", nn.Conv2d(3, inplanes, 3, 1, 1, bias=False)),
# #         ("bn1", nn.BatchNorm2d(num_features=inplanes)),
# #         ("relu1", nn.ReLU()),
# #         ("maxpool", nn.Identity()),
# #     ]

# #     arr_num_blocks = [2, 2, 2, 2]
# #     arr_dims = [32, 32, 16, 5]

# #     for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
# #         layer = resnet18._make_layer(
# #             torchvision.models.resnet.BasicBlock,
# #             dims,
# #             num_blocks,
# #             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
# #             stride=2 if i > 0 else 1,
# #             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
# #             dilate=False,
# #         )

# #         layers.append((f"layer{i+1}", layer))

# #     layers.extend(
# #         [
# #             ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
# #             ("flatten", nn.Flatten(start_dim=1)),
# #             ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
# #         ]
# #     )

# #     model = nn.Sequential(OrderedDict(layers))

# #     return model


# @register_model("resnet18compr1")
# def _resnet18imagenetc1(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=1, num_classes=num_classes, for_cifar=False
#     )


# @register_model("resnet18compr2")
# def _resnet18imagenetc2(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=2, num_classes=num_classes, for_cifar=False
#     )
