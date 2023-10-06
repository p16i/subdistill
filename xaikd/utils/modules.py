import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import batchnorm


# ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/batchnorm.py#L121
class Centering2D(batchnorm._BatchNorm):
    # This module is similar to BatchNorm except that it performs
    #  x = x - E[x],
    # without deviding the Var[x].
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        _, d, _, _ = input.shape
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            # Pat's change: we do NOT use self.running_var here.
            torch.ones(d).to(input.device),
            None,
            None,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class DiagonalScaling(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()

        self.scale = nn.Parameter(torch.randn(dims).reshape(1, dims, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dims).reshape(1, dims, 1, 1))

    def forward(self, x):
        return self.scale * x + self.bias
