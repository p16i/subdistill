import numpy as np
from numpy import typing as npt

import torch
from torch import nn
from torch.nn import functional as F

from xaikd.bases import OrthogonalBasis

from .register import register_policy
from .interface import LayerPolicy

from xaikd import utils


class SubtractMean(nn.Module):
    def __init__(self, mean: npt.NDArray, device) -> None:
        super().__init__()

        self.mean = torch.from_numpy(mean).reshape((1, -1, 1, 1)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean


@register_policy("basis-center-pca-learned-linearortho")
class OrthogonalPCAConvergenceCheckPolicy(LayerPolicy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__()

        k = student_dims
        d = teacher_dims

        self.basis = basis

        if layerwise_training:
            self.scaling_factor = 1
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )
        self.d = d
        self.k = k

        self.transformer_teacher_feats = SubtractMean(self.basis.mean, device)

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False),
            utils.modules.LinearOrtho(in_features=k, out_features=d),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats,
            transformed_teacher_feats,
            reduction="none",
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-center-pca-learned-linearortho-with-bias")
class OrthogonalPCAConvergenceCheckPolicy(OrthogonalPCAConvergenceCheckPolicy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(
            teacher_dims=teacher_dims,
            student_dims=student_dims,
            device=device,
            basis=basis,
            layerwise_training=layerwise_training,
        )

        k = student_dims
        d = teacher_dims

        self.transformer_student_feats = nn.Sequential(
            utils.modules.LinearOrtho(in_features=k, out_features=d, bias=True),
        ).to(device)
