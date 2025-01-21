import typing
import torch

import numpy as np


def collapse_dim(func, dim):

    def _collapse(x):

        return func(x).sum(dim=dim)

    return _collapse


def compute(
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """Compute Jacobian of output wrt input

    Args:
        output (torch.Tensor): 2-dimensional array (nb, d_out)
        input (torch.Tensor): array with size (nb, d_in, w, h)

    Returns:
        torch.Tensor: Jacboian with size (nb, d_out, d_in, w, h)
    """

    _, d_out = output.shape

    in_dims = input.shape
    assert len(in_dims) == 4

    grad_output = torch.zeros_like(output).to(input.device)

    arr_grads = []

    for dix in range(d_out):
        # refill the tensor with zero
        grad_output.zero_()

        grad_output[:, dix] = 1

        (grad,) = torch.autograd.grad(
            outputs=output, inputs=input, grad_outputs=grad_output, retain_graph=True
        )

        arr_grads.append(grad.detach())

    jacobian = torch.stack(arr_grads)

    # move `batch dimension` to the 1st dim
    jacobian = torch.swapdims(jacobian, 1, 0)

    return jacobian
