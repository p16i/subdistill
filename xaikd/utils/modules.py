from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import batchnorm
import numpy as np
from copy import deepcopy

import tempfile
import wandb
from scipy.stats import ortho_group


def has_batchnorm(model: nn.Module) -> bool:
    answer = False
    for m in model.children():
        if isinstance(m, nn.BatchNorm2d) or has_batchnorm(m):
            return True

    return answer


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
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
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


class Conv2dRotation(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.from_numpy(ortho_group.rvs(dim=dims)).float())

    def forward(self, x):
        return F.conv2d(x, self.weight.unsqueeze(2).unsqueeze(3))


def convert_bn_to_conv(bn: nn.BatchNorm2d) -> nn.Conv2d:
    assert bn.running_mean is not None
    bn_mean = bn.running_mean.clone()

    assert bn.running_var is not None
    bn_std = (bn.running_var.clone() + bn.eps) ** 0.5

    if bn.affine:
        bn_scale = bn.weight.clone()
        bn_shift = bn.bias.clone()
    else:
        bn_scale = 1
        bn_shift = 0

    W_bn = torch.diag(bn_scale / bn_std)

    b_bn = -(bn_scale / bn_std) * bn_mean + bn_shift

    d = b_bn.shape[0]

    conv_bn = nn.Conv2d(d, d, kernel_size=1)

    conv_bn.weight = nn.Parameter(W_bn.unsqueeze(2).unsqueeze(3))
    conv_bn.bias = nn.Parameter(b_bn)

    return conv_bn


def merge_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    conv_bn = convert_bn_to_conv(bn)

    return merge_convKxK_and_conv1x1(conv, conv_bn)


def merge_convKxK_and_conv1x1(convK: nn.Conv2d, conv1: nn.Conv2d) -> nn.Conv2d:
    np.testing.assert_allclose(np.array(conv1.padding), 0)
    np.testing.assert_allclose(conv1.kernel_size, 1)

    Wk = getattr(convK, "weight")
    if convK.bias is not None:
        bk = convK.bias
    else:
        bk = torch.zeros(size=(convK.out_channels,)).to(Wk.device)

    # shape: [out_channels, in_channels, 1, 1]
    W1 = getattr(conv1, "weight")
    # remove spatial dimensions
    W1 = W1.squeeze()
    b1 = getattr(conv1, "bias")

    # essentially, the new filter is the multiplication
    # of the Wk and W1 at every spatial location
    Wh = torch.einsum("jiwh,kj->kiwh", Wk, W1)

    bh = W1 @ bk + b1

    merged_conv = deepcopy(convK)
    merged_conv.weight = nn.Parameter(Wh)
    merged_conv.bias = nn.Parameter(bh)

    return merged_conv


def load_model_from_checkpoint(
    model_template_object: nn.Module,
    checkpoint_path: str,
    model_key: str,
    device="cpu",
) -> nn.Module:
    # todo: add test
    ckpt = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
        weights_only=False,
    )

    state_dict = ckpt["state_dict"]

    student_state_dict = dict()

    for key in state_dict.keys():
        if key.split(".")[0] == model_key:
            student_state_dict[key] = state_dict[key]

    trainer_wrapper = nn.Sequential(OrderedDict([(model_key, model_template_object)]))
    trainer_wrapper.load_state_dict(student_state_dict)

    return model_template_object


def load_model_from_wandb_artifact(
    model_template_object: nn.Module,
    run_path: str,
    model_key: str,
    wandb_artifact_suffix: str,
    device="cpu",
):
    # todo: add test
    agent = wandb.Api()

    slugs = run_path.split("/")
    wandb_project = "/".join(slugs[:2])
    wandb_runid = slugs[-1]

    artifact: wandb.Artifact = agent.artifact(
        f"{wandb_project}/model-{wandb_runid}:{wandb_artifact_suffix}"
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        artifact_dir = artifact.download(root=tmpdirname)
        model = load_model_from_checkpoint(
            model_template_object=model_template_object,
            checkpoint_path=f"{artifact_dir}/model.ckpt",
            model_key=model_key,
            device=device,
        )
        model.eval()
        return model
