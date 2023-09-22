import torch

import typing
from torch import nn


from torchvision import models

from . import register_model, interfaces


models.vgg.cfgs["VGG8"] = [
    64,
    "M",
    128,
    "M",
    256,
    "M",
    512,
    "M",
    512,
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

    # def split_at(self, layer: str):
    #     assert hasattr(self, layer)

    #     layer_ix = int(layer[-1]) - 1

    #     layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]

    #     layers_in_head = layers[:layer_ix] if layer_ix > 0 else []
    #     layers_in_classifier = layers[layer_ix + 1 :]

    #     layer_module = layers[layer_ix]

    #     head = nn.Sequential(*layers_in_head)

    #     classifier = nn.Sequential(
    #         *layers_in_classifier,
    #         self.avgpool,
    #         nn.Flatten(start_dim=1),
    #         self.classifier,
    #     )

    #     return head, layer_module, classifier

    @classmethod
    def cast(cls, model: models.vgg.VGG):
        assert isinstance(model, models.vgg.VGG)

        return DistillableVGG(model)


@register_model("vgg11")
def _vgg11(num_classes: int) -> nn.Module:
    model = models.vgg.vgg11(num_classes=num_classes)

    model.num_classes = num_classes

    return model


@register_model("cifar-vgg11")
def _cifar_vgg11(num_classes: int) -> nn.Module:
    return _vgg11(num_classes)


@register_model("vgg8")
def _vgg8(num_classes: int) -> nn.Module:
    model = models.vgg._vgg("VGG8", False, None, None, num_classes=num_classes)

    model.num_classes = num_classes

    return model
