import torch

from torch import nn

from torchvision.models import vgg

import numpy as np


class VGGFeatureBlocks(nn.Module):
    def __init__(self, model: vgg.VGG) -> None:
        super().__init__()

        assert isinstance(model, vgg.VGG)

        block_ix = 1
        curr_block = []
        for layer in model.features:
            curr_block.append(layer)

            if isinstance(layer, nn.MaxPool2d):
                setattr(self, f"layer{block_ix}", nn.Sequential(*curr_block))
                curr_block = []
                block_ix += 1

        self.avgpool = model.avgpool
        self.classifier = model.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)

        return x
