import pytest

import torch
from xaikd import models, utils
from xaikd.models.vgg import DistillableVGG

from torchvision import models as tvm
import numpy as np

import models as test_models


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


# @pytest.mark.skip
# @pytest.mark.parametrize("slug", ["cifar100-vgg11-p1", "imagenet-vgg16-tv"])
# @pytest.mark.parametrize("layer", ["layer1", "layer2", "layer3", "layer4", "layer5"])
# @pytest.mark.slow
# def test_split_model(slug, layer):
#     model = models.get_trained_model(slug)

#     head, layer_module, classifier = model.split_at(layer)
#     if "imagenet" in slug:
#         input = torch.randn(2, 3, 224, 224)
#     else:
#         input = torch.randn(2, 3, 32, 32)

#     assert layer_module == getattr(model, layer)

#     with torch.no_grad():
#         x = head(input)
#         x = layer_module(x)
#         actual = classifier(x).numpy()
#         expected = model(input).numpy()

#         np.testing.assert_allclose(actual, expected)


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
def test_split_vgg16_model(layer_ix):
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
def test_split_vgg19_model(layer_ix):
    test_models._test_split_model(
        "imagenet-vgg19bn-tv", f"features.{layer_ix}", models.vgg.split_model_at
    )
