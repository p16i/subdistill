import pytest

import numpy as np
import torch
from torchvision import transforms as T

from zennit.attribution import Gradient
from xaikd import models, lrp

pytest.skip(allow_module_level=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name", ["imagenet-mobilenetl-tv", "imagenet-mobilenets-tv"]
)
def test_callable(model_name):
    torch.manual_seed
    rng = torch.Generator().manual_seed(1)
    model = models.get_trained_model(model_name)

    lb = torch.zeros(3).reshape((1, -1, 1, 1))
    hb = torch.ones(3).reshape((1, -1, 1, 1))

    input = torch.randn(1, 3, 224, 224, generator=rng)
    with torch.no_grad():
        expected_output = model(input).numpy()

    assert hasattr(model, "features")

    with Gradient(
        model=model, composite=lrp.mobilenets._build_composite(lb=lb, hb=hb)
    ) as attributor:
        pass
        actual_output, hm = attributor.forward(input, lambda logits: logits)

        assert np.isfinite(actual_output.detach().numpy()).any()
        assert np.isfinite(hm.detach().numpy()).any()

        actual_output = actual_output.detach().numpy()

    np.testing.assert_allclose(actual_output, expected_output, atol=1e-1)
