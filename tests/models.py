import pytest

from xaikd import models


@pytest.mark.parametrize("slug", ["cifar10-resnet18-p1", "cifar100-resnet18-p1"])
def test_get_models(slug):
    assert models.get_model(slug) is not None
