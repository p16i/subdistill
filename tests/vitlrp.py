import pytest

import numpy as np
import torch

from zennit.attribution import Gradient
from xaikd import vitlrp
from xaikd import models


@pytest.mark.slow
def test_callable():
    vit = models.get_trained_model("imagenet-nfnetf0-dm")

    lb = torch.ones(3).reshape((1, -1, 1, 1))
    hb = torch.ones(3).reshape((1, -1, 1, 1))

    input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected_output = vit(input).numpy()

    with Gradient(
        model=vit, composite=vitlrp._build_composite(lb=lb, hb=hb)
    ) as attributor:
        pass
        actual_output, hm = attributor(input)

        assert np.isfinite(actual_output.detach().numpy()).any()
        assert np.isfinite(hm.detach().numpy()).any()

        actual_output = actual_output.detach().numpy()

    np.testing.assert_allclose(
        actual_output,
        expected_output,
    )
