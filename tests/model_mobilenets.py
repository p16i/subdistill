import pytest

import torch
import numpy as np

import models as test_models
import torchvision

from xaikd import models, utils

from torch.utils.data import TensorDataset, DataLoader


@pytest.mark.parametrize(
    "slug",
    ["imagenet-mobilenetl-tv", "imagenet-mobilenets-tv"],
)
@pytest.mark.slow
def test_get_model(slug):
    test_models._test_get_model(slug)


@pytest.mark.parametrize(
    "model_name",
    ["imagenet-mobilenetl-tv"],
)
@pytest.mark.parametrize("layer", [f"features.{i}" for i in range(17)])
# @pytest.mark.parametrize("layer", [f"features.10"])
@pytest.mark.slow
def test_split_model_large(model_name, layer):

    test_models._test_split_model(
        model_name, layer, models.mobilenets.split_model_at, atol=1e-5
    )


@pytest.mark.parametrize(
    "model_name",
    ["imagenet-mobilenets-tv"],
)
@pytest.mark.parametrize("layer", [f"features.{i}" for i in range(12)])
@pytest.mark.slow
def test_split_model_small(model_name, layer):

    test_models._test_split_model(
        model_name, layer, models.mobilenets.split_model_at, atol=1e-5
    )


@pytest.mark.parametrize(
    "model_name,num_layers",
    [
        ("imagenet-mobilenetl-tv", 17),
        ("imagenet-mobilenets-tv", 12),
    ],
)
def test_get_layer_dimensions(model_name, num_layers):
    arr_layers = [f"features.{i}" for i in range(num_layers)]
    model = models.get_trained_model(model_name)
    x = torch.randn((7, 3, 224, 224))
    y = torch.randint(low=0, high=20, size=(7,))

    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=32)

    utils.get_dimensions_at_layers(model, dl, arr_layers)
