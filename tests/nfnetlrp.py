import pytest

import numpy as np

import torch

from xaikd import models, nfnetlrp
from zennit.attribution import Gradient


@pytest.mark.slow
def test_callable():
    nfnet = models.get_trained_model("imagenet-nfnetf0-dm")

    lb = torch.ones(3).reshape((1, -1, 1, 1))
    hb = torch.ones(3).reshape((1, -1, 1, 1))

    input = torch.randn(1, 3, 224, 224)

    with Gradient(
        model=nfnet, composite=nfnetlrp.EpsilonGammaBox(lb=lb, hb=hb)
    ) as attributor:
        pass
        output, hm = attributor(input)

        assert np.isfinite(output.detach().numpy()).any()
        assert np.isfinite(hm.detach().numpy()).any()


# @todo: model w/o attribution  and attribution share the same output
