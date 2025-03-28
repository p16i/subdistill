import pytest

import torch
import numpy as np

from xaikd import models, utils

import models as test_models


@pytest.mark.parametrize(
    "slug",
    [
        "cifar100-resnet18-v1",
    ],
)
@pytest.mark.slow
def test_get_cifar100_model(slug):
    test_models._test_get_model(slug)


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
@pytest.mark.slow
def test_get_imagenet_model(slug):
    test_models._test_get_model(slug)


@pytest.mark.parametrize(
    "slug",
    [
        "celeba-resnet18-finetunedv1",
        "celeba-resnet50-finetunedv1",
        "celeba-wideresnet50_2-finetunedv1",
    ],
)
@pytest.mark.slow
def test_get_celeba_model(slug):
    test_models._test_get_model(slug)


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
