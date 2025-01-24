import pytest

import torch
from torch import nn

from xaikd import jacobian

import numpy as np


def test():
    x = torch.randn((32, 128, 8, 8))

    nb = x.shape[0]
    size_in = x.shape[1:]

    d_out = 16

    model = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=d_out, kernel_size=2),
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(start_dim=1),
    )
    model.eval()

    input = x.clone().requires_grad_(True)
    output: torch.Tensor = model(input)

    actual = jacobian.compute(output=output, input=input)

    assert actual.shape == (nb, d_out, *size_in)

    actual = actual.detach().numpy()

    expected = []

    for dix in range(d_out):

        _input = x.clone().requires_grad_(True)
        grad_output = torch.zeros((nb, d_out))
        grad_output[:, dix] = 1
        (grad,) = torch.autograd.grad(
            outputs=model(_input),
            inputs=_input,
            grad_outputs=grad_output,
            retain_graph=True,
        )

        expected.append(grad.detach().cpu().numpy())

    expected = np.array(expected)
    expected = np.swapaxes(expected, 1, 0)

    np.testing.assert_allclose(actual, expected)
