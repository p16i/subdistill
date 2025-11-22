import typing

import numpy as np
from numpy import typing as npt

import torch
from torch import nn
from torch.nn import functional as F

from xaikd.bases import OrthogonalBasis, Identity
from xaikd.bases.adapter import Adapter, AdapterMode
from xaikd import utils

from .register import register_policy
from .interface import LayerPolicy

from pytorch_lightning import LightningModule


@register_policy("nothing")
class NothingPolicy(LayerPolicy):
    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def forward(self, teacher_feats, student_feats):
        return torch.tensor(0.0).to(teacher_feats.device)

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:
        return torch.tensor(0.0).to(transformed_teacher_feats.device)


@register_policy("basis-center-rotationv2")
class OrthogonalBasisCenterRotationV2Policy(LayerPolicy):
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

        self.basis = basis

        if layerwise_training:
            self.scaling_factor = 1.0
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k),
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

        loss_mse = loss_mse / self.scaling_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-center-rotationv2-always-normalize")
class OrthogonalBasisCenterRotationV2AlwaysNormalizePolicy(
    OrthogonalBasisCenterRotationV2Policy
):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device, basis, layerwise_training)

        k = student_dims

        self.scaling_factor = 1.0

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=True).to(device),
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy("basis-center-rotationv3")
class OrthogonalBasisCenterRotationV3Policy(OrthogonalBasisCenterRotationV2Policy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device, basis, layerwise_training)

        k = student_dims

        self.transformer_student_feats = nn.Sequential(
            utils.modules.SubtractingMean().to(device),
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy("basis-bn-rotationv2")
class OrthogonalBasisBNRotationV2Policy(OrthogonalBasisCenterRotationV2Policy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device, basis, layerwise_training)

        k = student_dims

        self.transformer_student_feats = nn.Sequential(
            nn.BatchNorm2d(k),
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy("basis-bias-rotationv2")
class OrthogonalBasisBiasRotationV2Policy(OrthogonalBasisCenterRotationV2Policy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device, basis, layerwise_training)

        k = student_dims

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Bias(k=k),
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy("basis-rotation-bias")
class OrthogonalBasisRotationBiasV2Policy(OrthogonalBasisCenterRotationV2Policy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device, basis, layerwise_training)

        k = student_dims

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Rotate(k=k),
            utils.modules.Bias(k=k),
        ).to(device)


@register_policy("basis-center-rotationv2-always-normalizing")
class OrthogonalBasisCenterRotationV2AlwaysNormalizingPolicy(LayerPolicy):
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

        self.basis = basis

        self.scaling_factor = np.sum(self.basis.get_scale_factors_for_k(student_dims))

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k),
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

        loss_mse = loss_mse / self.scaling_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-center-ortho")
class OrthogonalBasisCenterOrthoPolicy(LayerPolicy):
    """
    This should be use with basis-identity
    """

    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__()

        d = teacher_dims
        k = student_dims

        assert isinstance(basis, (Identity,))

        self.basis = basis

        if layerwise_training:
            self.scaling_factor = 1.0
        else:
            # here, we use all the dimensions
            self.scaling_factor = np.sum(self.basis.get_scale_factors_for_k(d))

        # here, we use k=d because we don't want to do any projection
        self.transformer_teacher_feats = basis.construct_adapter(
            k=d, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False),
            utils.modules.LinearOrtho(in_features=k, out_features=d, bias=False),
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

        loss_mse = loss_mse / self.scaling_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-center-softortho")
class OrthogonalBasisCenterSoftOrthoPolicy(OrthogonalBasisCenterOrthoPolicy):
    """
    This should be use with basis-identity
    """

    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device, basis, layerwise_training)

        d = teacher_dims
        k = student_dims

        lin = nn.Conv2d(in_channels=k, out_channels=d, bias=False, kernel_size=1)

        W = lin.weight.squeeze()  # shape (out_features, in_features)

        U, _, _ = torch.linalg.svd(W, full_matrices=False)

        lin.weight.data = U.unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            print("Initialized soft-ortho layer with SVD.")
            svdvals = torch.linalg.svdvals(lin.weight.data.squeeze())
            print(
                f"sigvals: min={torch.min(svdvals).item()}, max={torch.max(svdvals).item()}"
            )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False), lin
        ).to(device)

    def additional_loss(self, module: LightningModule, prefix="") -> torch.Tensor:
        linear_layer: nn.Linear = self.transformer_student_feats[1]
        W = linear_layer.weight.squeeze()  # shape (out_features, in_features)

        _, _ = W.shape

        sigvals = torch.linalg.svdvals(W)  # shape (min(out_features, in_features),)

        module.log(f"{prefix}_softortho_sigvals_max", sigvals.max().item())
        module.log(f"{prefix}_softortho_sigvals_min", sigvals.min().item())
        # this is a stable way to implement ||W^T @ W - I_K ||2^2
        loss = torch.sum((sigvals - 1.0) ** 2)

        return 1000 * loss


@register_policy("basis-center-linear")
class OrthogonalBasisCenterLinearPolicy(OrthogonalBasisCenterOrthoPolicy):
    """
    This should be use with basis-identity
    """

    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        layerwise_training: bool,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device, basis, layerwise_training)

        d = teacher_dims
        k = student_dims

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False),
            nn.Conv2d(in_channels=k, out_channels=d, kernel_size=1, bias=False),
        ).to(device)


@register_policy("basis-center-rotationv2-no-normalization")
class OrthogonalBasisCenterRotationV2NoNormalizationPolicy(LayerPolicy):
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

        self.basis = basis

        self.scaling_factor = 1.0

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k),
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

        loss_mse = loss_mse / self.scaling_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-rotation")
class OrthogonalBasisRotationPolicy(LayerPolicy):
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

        self.basis = basis

        if layerwise_training:
            self.scaling_factor = 1.0
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Rotate(k=k),
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

        loss_mse = loss_mse / self.scaling_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse
