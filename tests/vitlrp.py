import pytest

import numpy as np
import torch

from zennit.attribution import Gradient
from xaikd import vitlrp
from xaikd import models


@pytest.mark.slow
def test_callable():
    torch.manual_seed
    rng = torch.Generator().manual_seed(1)
    vit = models.get_trained_model("imagenet-vitb-tv")

    lb = torch.ones(3).reshape((1, -1, 1, 1))
    hb = torch.ones(3).reshape((1, -1, 1, 1))

    input = torch.randn(1, 3, 224, 224, generator=rng)
    with torch.no_grad():
        expected_output = vit(input).numpy()

    assert hasattr(vit.encoder, "layers")

    with Gradient(
        model=vit, composite=vitlrp._build_composite(lb=lb, hb=hb)
    ) as attributor:
        pass
        actual_output, hm = attributor(input)

        assert np.isfinite(actual_output.detach().numpy()).any()
        assert np.isfinite(hm.detach().numpy()).any()

        actual_output = actual_output.detach().numpy()

    assert hasattr(vit.encoder, "layers")

    for helper_attribute in [
        "_ln",
        "_layers",
        "_dropout",
    ]:
        assert not hasattr(vit.encoder, helper_attribute)

    np.testing.assert_allclose(actual_output, expected_output, atol=1e-5)
