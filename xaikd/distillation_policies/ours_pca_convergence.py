import numpy as np
from numpy import typing as npt

from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F

from xaikd.bases import OrthogonalBasis

from .register import register_policy
from .interface import LayerPolicy, PolicyWithLogging
k
from xaikd import utils


class SubtractMean(nn.Module):
    def __init__(self, mean: npt.NDArray, device) -> None:
        super().__init__()

        self.mean = torch.from_numpy(mean).reshape((1, -1, 1, 1)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor
        return x - self.mean


@register_policy("basis-recon-check-with-linear")
class OrthogonalPCAConvergenceWithLinearPolicy(LayerPolicy, PolicyWithLogging):
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

        self.d = d
        self.k = k

        self.transformer_teacher_feats = SubtractMean(self.basis.mean, device)

        self.transformer_student_feats = nn.Linear(
            in_features=k, out_features=d, bias=True
        )

        self.Uk = torch.from_numpy(self.basis.get_Uk(k)).float().to(device)

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

    def log(
        self,
        module: LightningModule,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
        prefix: str,
    ):
        ref = self.transformer_student_feats(student_feat)
        Uk = self.Uk

        recon = F.conv2d(
            self.transformer_student_feats(student_feat),
            (Uk @ Uk.T).unsqueeze(2).unsqueeze(3),
        )

        err = torch.flatten((ref - recon) ** 2, start_dim=1).mean()

        module.log(f"{prefix}_recon_on_basis", err, on_epoch=True)
