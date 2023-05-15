import torch
import pytest
import numpy as np

from xaikd import models


@pytest.mark.parametrize(
    "slug", ["cifar10-resnet18-p1", "cifar100-resnet18-p1", "imagenet-resnet18-tv"]
)
def test_get_models(slug):
    model = models.get_model(slug)
    assert not model.training
    assert models.get_model(slug) is not None


@pytest.mark.parametrize(
    "slug", ["cifar10-resnet18-p1", "cifar100-resnet18-p1", "imagenet-resnet18-tv"]
)
@pytest.mark.parametrize("layer", ["layer1", "layer2", "layer3", "layer4"])
@pytest.mark.slow
def test_split_model(slug, layer):
    model = models.get_model(slug)

    head, classifier = models.resnet.spit_resnet_18_at(model, layer)

    if "imagenet" in slug:
        input = torch.randn(10, 3, 224, 224)
    else:
        input = torch.randn(10, 3, 32, 32)

    with torch.no_grad():
        actual = classifier(head(input)).numpy()
        expected = model(input).numpy()

        np.testing.assert_allclose(actual, expected)
