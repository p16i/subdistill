import typing

from collections import OrderedDict

import torch
from torch import nn
import numpy as np

from functools import partial


from xaikd import constants
from xaikd.utils.modules import (
    merge_conv_and_bn,
    merge_convKxK_and_conv1x1,
)


from . import (
    add_model_to_registry,
    vit_students,
    students_mobilenet,
    students_mobilenetv4,
)


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


class StudentModel(nn.Module):
    def __init__(self, arr_dims: typing.List[int], num_classes: int, **kwargs):
        super().__init__()

        assert len(arr_dims) == 4

        inplane = arr_dims[0]

        self.stem = nn.Sequential(
            ConvBN(in_channels=3, out_channels=arr_dims[0], kernel_size=3),
            nn.ReLU(),
            ConvBN(in_channels=arr_dims[0], out_channels=arr_dims[0], kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

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

            adapter = ConvBN(
                in_channels=layer_dim, out_channels=layer_dim, kernel_size=3
            )

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

            setattr(self, f"layer{lix+1}", layer)

            prev_dim = arr_dims[lix]

        last_d = arr_dims[-1]

        self.classifier = nn.Sequential(
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

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.classifier(x)

        return x


def canonize_student_model(model: StudentModel) -> nn.Module:
    assert isinstance(model, StudentModel)

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


class LeNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNetXL(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        return x


def _generate_model_function():

    add_model_to_registry("student-lenet", LeNet)
    add_model_to_registry("student-lenetxl", LeNetXL)

    for arr_dims in constants.ARR_STUDENT_DIMENSIONS:

        slug = "-".join(np.array(arr_dims).astype(str).tolist())

        add_model_to_registry(
            f"student-{slug}", partial(StudentModel, arr_dims=arr_dims)
        )


_generate_model_function()
