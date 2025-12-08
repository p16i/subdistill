import pytest
import torch
import numpy as np

from xaikd.explainersv2 import logit_gap_wrt_target


@torch.no_grad()
def test():
    x = torch.tensor(
        [
            [0.1, 0.5, 0.3],
            [0.2, 0.1, 0.4],
            [0.2, 0.5, 0.2],
        ]
    )

    y = torch.tensor([1, 2, 2])

    actual = logit_gap_wrt_target(x, y, num_classes=3).numpy()
    expected = np.array(
        [
            [0.0, 0.5, -0.3],
            [-0.2, 0.0, 0.4],
            [0.0, -0.5, 0.2],
        ]
    )

    np.testing.assert_allclose(actual, expected)
