import torch
from torch.nn import functional as F
import numpy as np

from xaikd.explainersv2 import logit_gap_wrt_target, LogitGapWrtTarget


@torch.no_grad()
def test_logit_gap_wrt_target():
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


def test_logit_gap_layer():
    layer = LogitGapWrtTarget(num_classes=3)

    x = torch.tensor(
        [
            [0.1, 0.5, 0.3],
            [0.2, 0.1, 0.4],
            [0.2, 0.5, 0.2],
        ]
    )

    x.requires_grad = True

    actual = layer(x)

    (
        actual * F.one_hot(torch.tensor([1, 2, 2]), num_classes=3).float()
    ).sum().backward()

    expected = np.array(
        [
            [0.1 - 0.5, 0.5 - 0.3, 0.3 - 0.5],
            [0.2 - 0.4, 0.1 - 0.4, 0.4 - 0.2],
            [0.2 - 0.5, 0.5 - 0.2, 0.2 - 0.5],
        ]
    )

    np.testing.assert_allclose(actual.detach().numpy(), expected)

    assert x.grad is not None

    np.testing.assert_allclose(
        x.grad.numpy(),
        np.array(
            [
                [0.0, 1.0, -1.0],
                [-1, 0.0, 1.0],
                [0.0, -1.0, 1.0],
            ]
        ),
    )


def test_logit_gap_degenerate():
    logits = torch.FloatTensor(
        [[-1.6193185, -2.5063117, -0.24580102, -4.5831337, -6.5929985]]
    )
    y = torch.LongTensor([0])

    output = logit_gap_wrt_target(logits, y, num_classes=5).numpy()

    expected = np.array(
        [
            [-1.6193185, 0, 0.24580102, 0, 0],
        ]
    )
    np.testing.assert_allclose(output, expected)
