import torch
import typing

from abc import ABC

import numpy as np

from torch import nn
from torch.nn import functional as F

from xaikd.bases import Basis, AdapterMode

from torch.nn.modules import batchnorm

from scipy.stats import ortho_group

LAYER_POLICY = dict()


class Policy(nn.Module, ABC):
    transformer_teacher_feats: nn.Module
    transformer_student_feats: nn.Module

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, teacher_feats, student_feats) -> torch.Tensor:
        transformed_teacher_feats = self.transformer_teacher_feats(teacher_feats)
        transformed_student_feats = self.transformer_student_feats(student_feats)

        assert transformed_student_feats.shape == transformed_teacher_feats.shape

        return self.criterion(transformed_teacher_feats, transformed_student_feats)


def register_layer_policy(name):
    """Decorator to register a layer policy"""

    def wrapped(fn):
        """Wrapped function to register a layer policy provider with`name`"""
        LAYER_POLICY[name] = fn

        return fn

    return wrapped


class Affine2D(nn.Module):
    def __init__(self, k: int, device: str) -> None:
        super().__init__()

        self.W = nn.Parameter(torch.randn(k)).to(device)
        self.b = nn.Parameter(torch.randn(k)).reshape((1, -1, 1, 1)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.conv2d(
                x,
                torch.diag(self.W).unsqueeze(2).unsqueeze(3),
            )
            + self.b
        )


class Centering(batchnorm._BatchNorm):
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


def get_layer_policy(name: str, **kwargs) -> Policy:
    return LAYER_POLICY[name](**kwargs)


class LayerPolicyCollection(nn.ModuleList):
    def __init__(
        self, layers: typing.List[str], policies: typing.Iterable[Policy]
    ) -> None:
        super().__init__(policies)

        assert len(layers) == len(policies)

        self.layers = layers
        self.policies = policies

    def forward(self, x):
        pass


class KLPolicy(Policy):
    # refs:
    # - https://github.com/yoshitomo-matsubara/torchdistill/blob/45ba679d4649512b520eb4ef7f97b757abf841ee/torchdistill/losses/mid_level.py#L100
    # - https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/KD.py

    def __init__(self, temperature: float):
        super().__init__()

        self.T = temperature
        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor
    ) -> torch.Tensor:
        kl = F.kl_div(
            torch.log_softmax(student_logits / self.T, dim=1),
            torch.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        )

        # remark: we muliply with scaling factor
        # as mentioned in Hinton et al. (2015) (paragraph before Section 2.1)
        return (self.T**2) * kl


@register_layer_policy("basis")
class OrthogonalBasisPolicy(Policy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: Basis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        # self.transformer_student_feats = Affine2D(k=k, device=device)
        self.transformer_student_feats = Centering(num_features=k, affine=False).to(
            device
        )

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("linteacher")
class LearnableAdapterTeacherPolicy(Policy):
    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_teacher_feats = nn.Conv2d(
            in_channels=teacher_dims,
            out_channels=student_dims,
            kernel_size=1,
        ).to(device)

        self.transformer_student_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("linstudent")
class LearnableAdapterStudentPolicy(Policy):
    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_student_feats = nn.Conv2d(
            out_channels=teacher_dims,
            in_channels=student_dims,
            kernel_size=1,
        ).to(device)

        self.transformer_teacher_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("vid")
class VIDPolicy(Policy):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    Adapted from https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/VID.py#L9C1-L54C20
    """

    # See Supplement  (P4330), the line before the last paragraph.
    init_pred_var = 5.0
    eps = 1e-5

    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                stride=stride,
            )

        # refs:
        # - https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L223
        # - https://github.com/yoshitomo-matsubara/torchdistill/blob/e615746172f653ebb59a29b396effba5aa766942/configs/sample/ilsvrc2012/vid/resnet18_from_resnet50.yaml#L123
        num_input_channels = student_dims
        num_mid_channel = num_target_channels = teacher_dims

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )

        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(self.init_pred_var - self.eps) - 1.0)
            * torch.ones(num_target_channels)
        )

        self.eps = self.eps

        self.transformer_student_feats = self.regressor
        self.transformer_teacher_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        target = transformed_teacher_feats
        pred_mean = transformed_student_feats

        # Paragraph after eq. 5
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)

        # eq. 6
        neg_log_prob = 0.5 * (
            (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )

        loss = torch.mean(neg_log_prob)

        return loss
