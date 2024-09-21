import numpy as np

import typing


from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    MobileNetV3,
)

import torch

from torch import nn


from . import interfaces
from . import register_model


def split_model_at(
    model: MobileNetV3, layer: str
) -> typing.Tuple[nn.Sequential, nn.Sequential]:
    assert isinstance(model, MobileNetV3)

    assert len(layer.split(".")) == 2

    layer_ix = int(layer.split(".")[-1])

    assert layer_ix >= 0

    layers = model.features

    # remark: we use zero-index; the head part therefore also includes with features.ix
    layers_in_head = layers[: layer_ix + 1]
    layers_in_classifier = layers[layer_ix + 1 :]

    head = nn.Sequential(*layers_in_head)

    classifier = nn.Sequential(
        *layers_in_classifier, model.avgpool, nn.Flatten(start_dim=1), model.classifier
    )

    return head, classifier


@register_model("imagenet-mobilenetl-tv")
def _mobilenetl() -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    setattr(model, "__last_layer", model.classifier[3])

    return model


@register_model("imagenet-mobilenets-tv")
def _mobilenets() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.num_classes = 1000

    setattr(model, "__last_layer", model.classifier[3])

    return model


# @register_model("imagenet-resnet34-tv")
# def _resnet34_imagenet() -> nn.Module:
#     model = torchvision.models.resnet34(weights=resnet.ResNet34_Weights.IMAGENET1K_V1)
#     model.num_classes = 1000

#     return model


# @register_model("imagenet-resnet50-tv")
# def _resnet50_imagenet() -> nn.Module:
#     model = torchvision.models.resnet50(weights=resnet.ResNet50_Weights.IMAGENET1K_V1)
#     model.num_classes = 1000

#     return model


# @register_model("imagenet-resnet101-tv")
# def _resnet101_imagenet() -> nn.Module:
#     model = torchvision.models.resnet101(weights=resnet.ResNet101_Weights.IMAGENET1K_V1)
#     model.num_classes = 1000

#     return model


# @register_model("imagenet-resnet152-tv")
# def _resnet152_imagenet() -> nn.Module:
#     model = torchvision.models.resnet152(weights=resnet.ResNet152_Weights.IMAGENET1K_V1)
#     model.num_classes = 1000

#     return model


# def _generate_resnet18_compressed(
#     compression_ratio: int,
#     num_classes: int,
#     for_cifar: bool,
#     parameterization_with="bn",
# ) -> nn.Module:
#     # todo: hard-corded everything for now.
#     resnet18 = torchvision.models.resnet.resnet18()

#     # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L184

#     inplanes = 32 // compression_ratio
#     # becuase inplance is modified throught the generation
#     # we have to reset attribute
#     resnet18.inplanes = inplanes

#     layers = []

#     if for_cifar:
#         # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
#         layers.extend(
#             [
#                 # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
#                 (
#                     "conv1",
#                     nn.Conv2d(
#                         3, inplanes, kernel_size=3, padding=1, stride=1, bias=False
#                     ),
#                 ),
#                 ("bn1", nn.BatchNorm2d(num_features=inplanes)),
#                 ("relu1", nn.ReLU()),
#                 ("maxpool", nn.Identity()),
#             ]
#         )
#     else:
#         layers.extend(
#             [
#                 # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L197
#                 (
#                     "conv1",
#                     nn.Conv2d(
#                         3, inplanes, kernel_size=7, padding=3, stride=2, bias=False
#                     ),
#                 ),
#                 ("bn1", nn.BatchNorm2d(num_features=inplanes)),
#                 ("relu1", nn.ReLU()),
#                 ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ]
#         )

#     arr_num_blocks = [2, 2, 2, 2]
#     arr_dims = [
#         32 // compression_ratio,
#         64 // compression_ratio,
#         64 // compression_ratio,
#         64 // compression_ratio,
#     ]

#     for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
#         layer = resnet18._make_layer(
#             torchvision.models.resnet.BasicBlock,
#             dims,
#             num_blocks,
#             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
#             stride=2 if i > 0 else 1,
#             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
#             dilate=False,
#         )

#         if parameterization_with == "bn":
#             parameterization_module = nn.BatchNorm2d(num_features=dims)
#         elif parameterization_with == "center":
#             parameterization_module = Centering2D(num_features=dims)
#         elif parameterization_with == "diag":
#             parameterization_module = DiagonalScaling(dims=dims)
#         elif parameterization_with == "lin":
#             parameterization_module = nn.Conv2d(
#                 in_channels=dims, out_channels=dims, kernel_size=1
#             )
#         elif parameterization_with == "rot":
#             parameterization_module = Conv2dRotation(dims=dims)
#         elif parameterization_with == "id":
#             parameterization_module = nn.Identity()
#         else:
#             raise ValueError(f"no `{parameterization_with}` available")

#         # # todo: this is temporary;
#         # todo: remove this after also in other derivative of the architecture.
#         layers.append(
#             (
#                 f"layer{i+1}",
#                 nn.Sequential(layer, parameterization_module),
#             )
#         )

#     layers.extend(
#         [
#             ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
#             ("flatten", nn.Flatten(start_dim=1)),
#             ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
#         ]
#     )

#     model = nn.Sequential(OrderedDict(layers))

#     return model


# @register_model("resnet18xscifarcompr1")
# def _cifarresnet18c1(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=1, num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18xscifarcompr1center")
# def _cifarresnet18c1center(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=1,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="center",
#     )


# @register_model("resnet18xscifarcompr1diag")
# def _cifarresnet18c1diag(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=1,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="diag",
#     )


# @register_model("resnet18xscifarcompr1lin")
# def _cifarresnet18c1lin(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=1,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="lin",
#     )


# @register_model("resnet18xscifarcompr1rot")
# def _cifarresnet18c1rot(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=1,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="rot",
#     )


# @register_model("resnet18xscifarcompr4rot")
# def _cifarresnet18c4rot(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=4,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="rot",
#     )


# @register_model("resnet18xscifarcompr1id")
# def _cifarresnet18c1id(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=1,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="id",
#     )


# @register_model("resnet18xscifarcompr2id")
# def _cifarresnet18c2id(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=2,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="id",
#     )


# @register_model("resnet18xscifarcompr4id")
# def _cifarresnet18c4id(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=4,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="id",
#     )


# @register_model("resnet18xscifarcompr2")
# def _cifarresnet18c2(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=2, num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18xscifarcompr4")
# def _cifarresnet18c4(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=4, num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18xscifarcompr2lin")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=2,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="lin",
#     )


# @register_model("resnet18xscifarcompr4lin")
# def _cifarresnet18c4lin(num_classes: int):
#     return _generate_resnet18_compressed(
#         compression_ratio=4,
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="lin",
#     )


# def _generate_resnet18_manual_block(
#     arr_dims: typing.List[int],
#     num_classes: int,
#     for_cifar: bool,
#     parameterization_with="bn",
# ) -> nn.Module:
#     # todo: hard-corded everything for now.
#     resnet18 = torchvision.models.resnet.resnet18()

#     # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L184

#     inplanes = arr_dims[0]
#     # becuase inplance is modified throught the generation
#     # we have to reset attribute
#     resnet18.inplanes = inplanes

#     layers = []

#     if for_cifar:
#         # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L4
#         layers.extend(
#             [
#                 # ref: https://github.com/lightly-ai/lightly/blob/b69b8b14c29121422479f23078488efca734a995/lightly/models/resnet.py#L193
#                 (
#                     "conv1",
#                     nn.Conv2d(
#                         3, inplanes, kernel_size=3, padding=1, stride=1, bias=False
#                     ),
#                 ),
#                 ("bn1", nn.BatchNorm2d(num_features=inplanes)),
#                 ("relu1", nn.ReLU()),
#                 ("maxpool", nn.Identity()),
#             ]
#         )
#     else:
#         layers.extend(
#             [
#                 # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L197
#                 (
#                     "conv1",
#                     nn.Conv2d(
#                         3, inplanes, kernel_size=7, padding=3, stride=2, bias=False
#                     ),
#                 ),
#                 ("bn1", nn.BatchNorm2d(num_features=inplanes)),
#                 ("relu1", nn.ReLU()),
#                 ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#             ]
#         )

#     arr_num_blocks = [2, 2, 2, 2]

#     for i, (dims, num_blocks) in enumerate(zip(arr_dims, arr_num_blocks)):
#         layer = resnet18._make_layer(
#             torchvision.models.resnet.BasicBlock,
#             dims,
#             num_blocks,
#             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L202
#             stride=2 if i > 0 else 1,
#             # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L78
#             dilate=False,
#         )

#         if parameterization_with == "bn":
#             parameterization_module = nn.BatchNorm2d(num_features=dims)
#         elif parameterization_with == "linnob":
#             parameterization_module = nn.Conv2d(
#                 in_channels=dims, out_channels=dims, kernel_size=1, bias=False
#             )
#         elif parameterization_with == "id":
#             parameterization_module = nn.Identity()
#         else:
#             raise

#         # # todo: this is temporary;
#         # todo: remove this after also in other derivative of the architecture.
#         layers.append(
#             (
#                 f"layer{i+1}",
#                 nn.Sequential(layer, parameterization_module),
#             )
#         )

#     layers.extend(
#         [
#             ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
#             ("flatten", nn.Flatten(start_dim=1)),
#             ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
#         ]
#     )

#     model = nn.Sequential(OrderedDict(layers))

#     return model


# @register_model("resnet18dims32-24-24-5")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[32, 24, 24, 5], num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18dims32-24-24-5-linnob")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[32, 24, 24, 5],
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="linnob",
#     )


# @register_model("resnet18dims32-24-24-5-id")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[32, 24, 24, 5],
#         num_classes=num_classes,
#         for_cifar=True,
#         parameterization_with="id",
#     )


# @register_model("resnet18dims16-8-8-5")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[16, 8, 8, 5], num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18dims8-16-16-5")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[8, 16, 16, 5], num_classes=num_classes, for_cifar=True
#     )


# @register_model("resnet18dims24-16-16-5")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[24, 16, 16, 5], num_classes=num_classes, for_cifar=True
#     )


# # imagenet
# # @register_model("resnet18dims32-24-24-10")
# # def _cifarresnet18c2lin(num_classes: int):
# #     return _generate_resnet18_manual_block(
# #         arr_dims=[32, 24, 24, 10], num_classes=num_classes, for_cifar=False
# #     )

# # @register_model("resnet18dims32-24-24-10")
# # def _cifarresnet18c2lin(num_classes: int):
# #     return _generate_resnet18_manual_block(
# #         arr_dims=[32, 24, 24, 10], num_classes=num_classes, for_cifar=False
# #     )


# @register_model("resnet18dims64-48-48-10")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[64, 48, 48, 10], num_classes=num_classes, for_cifar=False
#     )


# @register_model("resnet18dims48-32-32-10")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[48, 32, 32, 10], num_classes=num_classes, for_cifar=False
#     )


# @register_model("resnet18dims32-16-16-10")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[32, 16, 16, 10], num_classes=num_classes, for_cifar=False
#     )


# @register_model("resnet18dims16-8-8-10")
# def _cifarresnet18c2lin(num_classes: int):
#     return _generate_resnet18_manual_block(
#         arr_dims=[16, 8, 8, 10], num_classes=num_classes, for_cifar=False
#     )


# @register_model("resnet18")
# def _cifarresnet18c2lin(num_classes: int):
#     return resnet.resnet18(num_classes=num_classes)
