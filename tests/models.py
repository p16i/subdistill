import pytest

from xaikd import models


@pytest.mark.parametrize(
    "slug", ["cifar10-resnet18-p1", "cifar100-resnet18-p1", "imagenet-resnet18-tv"]
)
def test_get_models(slug):
    model = models.get_model(slug)
    assert not model.training
    assert models.get_model(slug) is not None