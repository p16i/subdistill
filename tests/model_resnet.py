import pytest

import torch
import numpy as np

from xaikd import models, utils

import models as test_models


@pytest.mark.parametrize(
    "slug",
    [
        "cifar100-resnet18-v1",
        "imagenet-resnet18-tv",
        "imagenet-resnet34-tv",
        "imagenet-resnet50-tv",
        "imagenet-resnet101-tv",
        "imagenet-resnet152-tv",
    ],
)
@pytest.mark.parametrize("layer", ["layer1", "layer2", "layer3", "layer4"])
@pytest.mark.slow
def test_split_resnet_model(slug, layer):
    test_models._test_split_model(slug, layer, models.resnet.split_model_at)
