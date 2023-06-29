import typing

import torch
from torch import nn

ARCH_LAYER_DIMENSIONS = dict(
    dict(
        resnet18={
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
            "layer4.0": 512,
            "layer4.1": 512,
        }
    )
)


def split_resnet_18_at(
    model: nn.Module, layer: str
) -> typing.Tuple[nn.Module, nn.Module, nn.Module]:

    assert len(layer.split(".")) == 1

    layer_ix = int(layer[-1]) - 1

    layers = [model.layer1, model.layer2, model.layer3, model.layer4]

    layers_in_head = layers[:layer_ix] if layer_ix > 0 else []
    layers_in_classifier = layers[layer_ix + 1 :]

    head = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool, *layers_in_head
    )

    layer_module = layers[layer_ix]

    classifier = nn.Sequential(
        *layers_in_classifier, model.avgpool, nn.Flatten(start_dim=1), model.fc
    )

    return head, layer_module, classifier
