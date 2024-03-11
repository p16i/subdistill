import pytest

import torch
import numpy as np

from xaikd import models, utils


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
    device = utils.get_device()

    model = models.get_trained_model(slug)
    model.to(device)

    head, classifier = models.resnet.split_model_at(model, layer)
    if "imagenet" in slug:
        input = torch.randn(10, 3, 224, 224)
    else:
        input = torch.randn(10, 3, 32, 32)
    input = input.to(device)

    with torch.no_grad():
        actual = classifier(head(input)).cpu().numpy()
        expected = model(input).cpu().numpy()

        np.testing.assert_allclose(actual, expected)
