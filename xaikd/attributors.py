import torch

from torchvision import models

from zennit.torchvision import ResNetCanonizer
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient


def make_attributor_for(model, input_transform) -> Gradient:
    # remark this only works for cifar10 and cifar100 for now
    assert type(model) == models.resnet.ResNet

    low, high = input_transform(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))

    canonizer = ResNetCanonizer()

    composite = EpsilonGammaBox(low=low, high=high, canonizers=[canonizer])

    return Gradient(model=model, composite=composite)
