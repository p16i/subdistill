import torch
import pytest
import numpy as np

from xaikd import models


@pytest.mark.parametrize(
    "slug",
    [
        "cifar10-resnet18-p1",
        "cifar100-resnet18-p1",
        "imagenet-resnet18-tv",
        "cifar100-vgg11-p1",
    ],
)
def test_get_models(slug):
    model = models.get_trained_model(slug)
    assert not model.training
    assert model is not None
    assert getattr(model, "__name") == slug
    assert len(getattr(model, "__layer_dimension").keys()) > 0

    # todo: check num class


# todo: che
