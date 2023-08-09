import typing

import types

from torch import nn

from torchvision.models import resnet

from . import interfaces


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
