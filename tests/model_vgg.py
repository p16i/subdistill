import pytest

import torch
from xaikd import models, utils

from torchvision import models as tvm
import numpy as np

import models as test_models

pytest.skip(reason="obsolte", allow_module_level=True)


@pytest.mark.parametrize(
    "slug",
    [
        "imagenet-vgg11-tv",
        "imagenet-vgg11bn-tv",
        "imagenet-vgg13-tv",
        "imagenet-vgg13bn-tv",
        "imagenet-vgg16-tv",
        "imagenet-vgg16bn-tv",
    ],
)
@pytest.mark.slow
def test_get_model(slug):
    test_models._test_get_model(slug)


@torch.no_grad()
@pytest.mark.parametrize(
    "arch_cls,num_classes,input_size",
    [
        (tvm.vgg11, 100, (10, 3, 32, 32)),
        (tvm.vgg16, 1000, (10, 3, 224, 224)),
    ],
)
@pytest.mark.skip
def test_group_feature_layers_vgg11(arch_cls, num_classes, input_size):
    model = arch_cls(num_classes=num_classes)
    model.eval()

    models_with_block = DistillableVGG(model)

    x = torch.randn(input_size)

    np.testing.assert_equal(
        utils.count_params_in_model(model),
        utils.count_params_in_model(models_with_block),
    )
    np.testing.assert_allclose(model(x), models_with_block(x))


@pytest.mark.parametrize("layer_ix", [5, 10, 15, 20])
@pytest.mark.slow
def test_split_vgg11_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg11-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )


@pytest.mark.parametrize("layer_ix", [7, 14, 21, 28])
@pytest.mark.slow
def test_split_vgg11bn_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg11bn-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )


@pytest.mark.parametrize("layer_ix", [9, 14, 19, 24])
@pytest.mark.slow
def test_split_vgg13_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg13-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )


@pytest.mark.parametrize("layer_ix", [13, 20, 27, 34])
@pytest.mark.slow
def test_split_vgg13bn_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg13bn-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )


@pytest.mark.parametrize("layer_ix", [9, 16, 23, 30])
@pytest.mark.slow
def test_split_vgg16_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg16-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )


@pytest.mark.parametrize("layer_ix", [13, 23, 33, 43])
@pytest.mark.slow
def test_split_vgg16bn_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg16bn-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )


@pytest.mark.parametrize("layer_ix", [9, 18, 27, 36])
@pytest.mark.slow
def test_split_vgg19_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg19-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )


@pytest.mark.parametrize("layer_ix", [13, 26, 39, 54])
@pytest.mark.slow
def test_split_vgg19bn_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg19bn-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )
