import pytest

import torch
import numpy as np

import models as test_models
import torchvision

from xaikd import models, utils

from torch.utils.data import TensorDataset, DataLoader


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

    test_models._test_split_model(
        model_name, layer, models.vit.split_model_at, atol=1e-5
    )


@pytest.mark.parametrize(
    "slug",
    ["imagenet-vitb-tv"],
)
@pytest.mark.slow
def test_get_model(slug):
    test_models._test_get_model(slug)


def test_get_layer_dimensions():
    arr_layers = [f"encoder.layers.{i}" for i in [8, 11]]
    model = models.get_trained_model("imagenet-vitb-tv")
    x = torch.randn((7, 3, 224, 224))
    y = torch.randint(low=0, high=20, size=(7,))

    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=32)

    utils.get_dimensions_at_layers(model, dl, arr_layers)


@pytest.mark.parametrize("lix", np.arange(12))
def test_intercept_module(lix):
    layer = f"encoder.layers.{lix}"
    model = models.get_trained_model("imagenet-vitb-tv")

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=False, detach_output=False
        )
    finally:
        hook.remove()

    assert isinstance(module, models.vit.ConvertTensorfromViTToCNNLikeShape)
