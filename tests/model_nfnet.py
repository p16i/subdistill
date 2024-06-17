import pytest

import torch
import numpy as np


import models as test_models

from xaikd import models, utils


@pytest.mark.parametrize(
    "slug",
    ["imagenet-nfnetf0-dm"],
)
@pytest.mark.slow
def test_get_model(slug):
    test_models._test_get_model(slug)


@pytest.mark.parametrize(
    "slug",
    ["imagenet-nfnetf0-dm"],
)
@pytest.mark.parametrize("layer", ["stages.0", "stages.1", "stages.2", "stages.3"])
@pytest.mark.slow
def test_split_nfnets_model(slug, layer):

    test_models._test_split_model(slug, layer, models.nfnet.split_model_at)
