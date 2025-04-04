import pytest

import numpy as np

import torch

from xaikd import models, lrp
from zennit.attribution import Gradient

pytest.skip(allow_module_level=True)


@pytest.mark.slow
def test_callable():
    nfnet = models.get_trained_model("imagenet-nfnetf0-dm")

    lb = torch.ones(3).reshape((1, -1, 1, 1))
    hb = torch.ones(3).reshape((1, -1, 1, 1))

    rng = torch.Generator()
    rng.manual_seed(1)
    input = torch.randn(1, 3, 224, 224, generator=rng)

    expected_output = nfnet(input).detach().numpy()

    with Gradient(
        model=nfnet, composite=lrp.nfnets._build_composite(lb=lb, hb=hb)
    ) as attributor:
        pass
        actual_output, hm = attributor.forward(input, lambda logits: logits)
        actual_output = actual_output.detach().numpy()

        assert np.isfinite(actual_output).any()
        assert np.isfinite(hm.detach().numpy()).any()

    np.testing.assert_allclose(actual_output, expected_output, atol=1e-6)
