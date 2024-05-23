import torch

from collections import OrderedDict

import typing
from torch import nn


from torchvision import models

from . import register_model, interfaces


from xaikd.utils.modules import (
    Centering2D,
    DiagonalScaling,
    merge_conv_and_bn,
    merge_convKxK_and_conv1x1,
)


@register_model("imagenet-vgg11-tv")
def _vgg11_imagenet() -> nn.Module:
    model = models.vgg11(weights=models.vgg.VGG11_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg11bn-tv")
def _vgg11bn_imagenet() -> nn.Module:
    model = models.vgg11_bn(weights=models.vgg.VGG11_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg13-tv")
def _vgg13_imagenet() -> nn.Module:
    model = models.vgg13(weights=models.vgg.VGG13_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg13bn-tv")
def _vgg13bn_imagenet() -> nn.Module:
    model = models.vgg13_bn(weights=models.vgg.VGG13_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg16-tv")
def _vgg16_imagenet() -> nn.Module:
    model = models.vgg16(weights=models.vgg.VGG16_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg16bn-tv")
def _vgg16bn_imagenet() -> nn.Module:
    model = models.vgg16_bn(weights=models.vgg.VGG16_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg19-tv")
def _vgg19_imagenet() -> nn.Module:
    model = models.vgg19(weights=models.vgg.VGG19_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


@register_model("imagenet-vgg19bn-tv")
def _vgg19bn_imagenet() -> nn.Module:
    model = models.vgg19_bn(weights=models.vgg.VGG19_BN_Weights.IMAGENET1K_V1)
    model.num_classes = 1000
    return model


models.vgg.cfgs["VGG8"] = [
    64,
    "M",
    64,
    "M",
    48,
    "M",
    32,
    "M",
    32,
    "M",
]


class DistillableVGG(interfaces.DistillableModel):
    # def __init__(self, model: models.vgg.VGG) -> None:
    #     super().__init__()

    #     assert isinstance(model, models.vgg.VGG)

    #     block_ix = 1
    #     curr_block = []
    #     for layer in model.features:
    #         curr_block.append(layer)

    #         if isinstance(layer, nn.MaxPool2d):
    #             setattr(self, f"layer{block_ix}", nn.Sequential(*curr_block))
    #             curr_block = []
    #             block_ix += 1

    #     self.avgpool = model.avgpool
    #     self.classifier = model.classifier

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = self.layer5(x)
    #     x = self.avgpool(x)
    #     x = x.flatten(start_dim=1)
    #     x = self.classifier(x)

    #     return x

    @classmethod
    def cast(cls, model: models.vgg.VGG):
        assert isinstance(model, models.vgg.VGG)

        return DistillableVGG(model)


def split_model_at(
    model: models.VGG, layer: str
) -> typing.Tuple[nn.Sequential, nn.Sequential]:
    assert isinstance(model, models.VGG)

    # layer in format `features.k``
    layer_ix = int(layer.split(".")[1])

    layers_in_head = model.features[: layer_ix + 1]
    layers_in_classifier = model.features[layer_ix + 1 :]

    head = nn.Sequential(*layers_in_head)

    classifier = nn.Sequential(
        *layers_in_classifier,
        model.avgpool,
        nn.Flatten(start_dim=1),
        model.classifier,
    )

    return head, classifier


@register_model("vgg11")
def _vgg11(num_classes: int) -> nn.Module:
    model = models.vgg.vgg11(num_classes=num_classes)

    model.num_classes = num_classes

    return model


@register_model("cifar-vgg11")
def _cifar_vgg11(num_classes: int) -> nn.Module:
    return _vgg11(num_classes)


@register_model("vgg8xs")
def _vgg8(num_classes: int, parameterization="bn") -> nn.Module:
    model = models.vgg._vgg("VGG8", False, None, None, num_classes=num_classes)

    dims = [48, 32, 32]
    layer_indices = [8, 11, 14]

    for lix, d in zip(layer_indices, dims):
        module = model.features[lix]
        assert isinstance(module, nn.MaxPool2d)

        if parameterization == "bn":
            parameterization_module = nn.BatchNorm2d(d)
        elif parameterization == "center":
            parameterization_module = Centering2D(num_features=d)
        elif parameterization == "diag":
            parameterization_module = DiagonalScaling(dims=d)
        elif parameterization == "lin":
            parameterization_module = nn.Conv2d(
                in_channels=d, out_channels=d, kernel_size=1
            )
        else:
            raise ValueError(f"No `{parameterization}` available!")

        model.features[lix] = nn.Sequential(module, parameterization_module)

    model.classifier = nn.Sequential(
        nn.Linear(dims[-1] * 7 * 7, 16),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(16, 16),
        nn.ReLU(True),
        nn.Dropout(p=0.5),
        nn.Linear(16, num_classes),
    )

    model.num_classes = num_classes

    return model


@register_model("vgg8xscenter")
def _vgg8lin(num_classes: int) -> nn.Module:
    return _vgg8(num_classes=num_classes, parameterization="center")


@register_model("vgg8xslin")
def _vgg8lin(num_classes: int) -> nn.Module:
    return _vgg8(num_classes=num_classes, parameterization="lin")


@register_model("vgg8xsdiag")
def _vgg8diag(num_classes: int) -> nn.Module:
    return _vgg8(num_classes=num_classes, parameterization="diag")


def _build_model(arr_dims: typing.List[int], num_classes: int) -> nn.Sequential:
    assert len(arr_dims) == 4

    inplane = arr_dims[0]

    layers = []

    stem = nn.Sequential(
        nn.Conv2d(3, arr_dims[0], kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(num_features=inplane),
        nn.ReLU(),
    )

    layers.append(("stem", stem))

    prev_dim = inplane

    for lix in range(len(arr_dims)):
        layer_dim = arr_dims[lix]
        layer = nn.Sequential(
            nn.Conv2d(
                prev_dim, prev_dim, kernel_size=1, padding=1, stride=1, bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                prev_dim, layer_dim, kernel_size=3, padding=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(
                num_features=layer_dim,
            ),
            nn.ReLU(),
            nn.Conv2d(
                layer_dim, layer_dim, kernel_size=3, padding=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(
                num_features=layer_dim,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # this is for adapting
            nn.Conv2d(
                layer_dim, layer_dim, kernel_size=3, padding=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(num_features=layer_dim),
        )

        prev_dim = arr_dims[lix]
        layers.append((f"layer{lix+1}", layer))

    layers.extend(
        [
            ("avgpool", nn.AdaptiveAvgPool2d((1, 1))),
            ("flatten", nn.Flatten(start_dim=1)),
            ("fc", nn.Linear(in_features=arr_dims[-1], out_features=num_classes)),
        ]
    )

    model = nn.Sequential(OrderedDict(layers))

    return model


class ConvBN(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x: torch.Tensor):
        return self.bn(self.conv(x))

    def canonize(self) -> nn.Conv2d:
        return merge_conv_and_bn(self.conv, self.bn)


@register_model("vggcustomdims-32-24-24-5")
def _vgg8diag(num_classes: int) -> nn.Module:
    return _build_model(arr_dims=[32, 24, 24, 5], num_classes=num_classes)


def _build_model_imagenet(
    arr_dims: typing.List[int], num_classes: int
) -> nn.Sequential:
    assert len(arr_dims) == 4

    inplane = arr_dims[0]

    layers = []

    stem = nn.Sequential(
        ConvBN(in_channels=3, out_channels=arr_dims[0], kernel_size=3),
        nn.ReLU(),
        ConvBN(in_channels=arr_dims[0], out_channels=arr_dims[0], kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    )

    layers.append(("stem", stem))

    prev_dim = inplane

    for lix in range(len(arr_dims)):
        layer_dim = arr_dims[lix]
        if lix == 0:
            _layer = []
        else:
            _layer = [
                nn.Conv2d(
                    in_channels=prev_dim,
                    out_channels=prev_dim,
                    kernel_size=1,
                    padding=0,
                ),
                nn.ReLU(),
            ]

        adapter = ConvBN(in_channels=layer_dim, out_channels=layer_dim, kernel_size=3)

        layer = nn.Sequential(
            *_layer,
            ConvBN(in_channels=prev_dim, out_channels=layer_dim, kernel_size=3),
            nn.ReLU(),
            ConvBN(in_channels=layer_dim, out_channels=layer_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # this is for adapting
            adapter,
        )

        prev_dim = arr_dims[lix]
        layers.append((f"layer{lix+1}", layer))

    last_d = arr_dims[-1]

    classifier = nn.Sequential(
        nn.Conv2d(
            in_channels=prev_dim,
            out_channels=prev_dim,
            kernel_size=1,
            padding=0,
        ),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(start_dim=1),
        nn.Linear(in_features=7 * 7 * last_d, out_features=last_d),
        nn.ReLU(),
        nn.Linear(in_features=last_d, out_features=last_d),
        nn.ReLU(),
        nn.Linear(in_features=last_d, out_features=num_classes),
    )

    layers.append(("classifier", classifier))

    model = nn.Sequential(OrderedDict(layers))

    # todo: we should do better here.
    setattr(model, "__layer_before_avgpool", "classifier.1")
    setattr(model, "__last_layer", "classifier.-1")

    return model


@register_model("vggcustomimagenetdims-32-24-24-10")
def _vgg8diag(num_classes: int) -> nn.Module:
    return _build_model_imagenet(arr_dims=[32, 24, 24, 10], num_classes=num_classes)


def canonize_model(model: nn.Module) -> nn.Module:
    # at the moment, this is for vggcustomimagenetdims
    features = []

    for layer_name in ["stem", "layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for modul in layer.children():
            if hasattr(modul, "canonize"):
                features.append(modul.canonize())
            else:
                features.append(modul)

    merged_features = []

    arr_cls_modules = list(model.classifier.children())

    last_adapter = features[-1]

    features = features[:-1]

    for fix in range(len(features)):
        modul = features[fix]
        if fix < len(features) - 2:
            next_modul = features[fix + 1]
        else:
            next_modul = None

        if (isinstance(modul, nn.Conv2d) and modul.kernel_size[0] > 1) and (
            isinstance(next_modul, nn.Conv2d) and next_modul.kernel_size[0] == 1
        ):
            merged_features.append(merge_convKxK_and_conv1x1(modul, next_modul))
        elif isinstance(modul, nn.Conv2d) and modul.kernel_size[0] == 1:
            continue
        else:
            merged_features.append(modul)

    arr_cls_modules = list(model.classifier.children())

    merged_conv = merge_convKxK_and_conv1x1(last_adapter, arr_cls_modules[0])

    return nn.Sequential(
        OrderedDict(
            [
                ("features", nn.Sequential(*merged_features)),
                ("classifier", nn.Sequential(merged_conv, *arr_cls_modules[1:])),
            ]
        )
    )
