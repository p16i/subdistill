import pytest
import torch
import numpy as np

from xaikd import attributors, models
from xaikd import utils

from torch.utils.data import DataLoader, Subset

from xaikd import datasets


@pytest.mark.parametrize(
    "arch,expected",
    [
        (
            "imagenet-resnet18-tv",
            [
                attributors.ResNetCanonizer,
            ],
        ),
        ("imagenet-vgg16-tv", []),
        (
            "imagenet-vgg16bn-tv",
            [
                attributors.VGGCanonizer,
            ],
        ),
    ],
)
def test_correct_canonizer(arch, expected):
    model = models.get_trained_model(arch)
    hb = torch.ones(3).reshape(1, -1, 1, 1)
    lb = -hb

    composite = attributors.get_arch_specific_composite(model, lb=lb, hb=hb)

    canonizers = composite.canonizers

    assert len(canonizers) == len(expected)
    for canonizer, type in zip(canonizers, expected):
        assert isinstance(canonizer, type)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "arch",
    [
        "imagenet-vgg11-tv",
        "imagenet-vgg11bn-tv",
        "imagenet-vgg16-tv",
        "imagenet-vgg16bn-tv",
        "imagenet-resnet18-tv",
        "imagenet-resnet34-tv",
    ],
)
def test_model_attributable(arch):
    torch.manual_seed(1)
    device = utils.get_device()

    model = models.get_trained_model(arch).to(device)

    data = torch.rand(5, 3, 224, 224)

    data = data.to(device)

    with attributors.make_attributor_for(
        model, input_statistics=[[0, 0, 0], [1, 1, 1]]
    ) as attributor:
        logits, attribution = attributor.forward(data, lambda logits: logits)

        assert not torch.isnan(logits).any()
        assert not torch.isnan(attribution).any()

    assert True
