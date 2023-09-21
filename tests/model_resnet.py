import pytest

import torch
import numpy as np

from xaikd import models


@pytest.mark.parametrize(
    "slug", ["cifar10-resnet18-p1", "cifar100-resnet18-p1", "imagenet-resnet18-tv"]
)
@pytest.mark.parametrize("layer", ["layer1", "layer2", "layer3", "layer4"])
@pytest.mark.slow
def test_split_resnet_model(slug, layer):
    model = models.get_trained_model(slug)

    head, layer_module, classifier = model.split_at(layer)
    if "imagenet" in slug:
        input = torch.randn(10, 3, 224, 224)
    else:
        input = torch.randn(10, 3, 32, 32)

    assert layer_module == getattr(model, layer)

    with torch.no_grad():
        x = head(input)
        x = layer_module(x)
        actual = classifier(x).numpy()
        expected = model(input).numpy()

        np.testing.assert_allclose(actual, expected)
