import pytest

import models as test_models
from xaikd import models


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
