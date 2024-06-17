import pytest

import torch
import numpy as np

import models as test_models
import torchvision

from xaikd import models


@torch.no_grad()
def test_vit_with_pre_post_transform():
    trng = torch.Generator()
    trng.manual_seed(1)

    inp = torch.rand((7, 3, 224, 224), generator=trng)
    model = models.get_trained_model("imagenet-vitb-tv")

    orig_model = torchvision.models.vit_b_16(weights=models.vit.ViT_B_16_Weights)
    orig_model.eval()

    expected = orig_model(inp)

    actual = model(inp)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "model_name",
    ["imagenet-vitb-tv"],
)
@pytest.mark.parametrize("layer", [f"encoders.layers.{i}" for i in range(12)])
@pytest.mark.slow
def test_split_resnet_model(model_name, layer):

    test_models._test_split_model(model_name, layer, models.vit.split_model_at)


@pytest.mark.parametrize(
    "slug",
    ["imagenet-vitb-tv"],
)
@pytest.mark.slow
def test_get_model(slug):
    test_models._test_get_model(slug)
