import typing

import numpy as np
from numpy import typing as npt

import torch
from torch import nn
from torch.nn import functional as F

from xaikd.bases import OrthogonalBasis
from xaikd.bases.adapter import Adapter, AdapterMode
from xaikd import utils

from .register import register_policy
from .interface import LayerPolicy


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


@register_policy("basis-bn-sum-normalized")
class OrthogonalBasisBatchNormSumNormalizedPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        if layerwise_training:
            # fixme: add tests
            self.scaling_factor = 1
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )

        print(
            f"basis-bn (teacher_dim={teacher_dims}); scaling factor={self.scaling_factor} "
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=True, affine=True
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.scaling_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-bn-sum-normalized-always")
class OrthogonalBasisBatchNormSumNormalizedPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.scaling_factor = np.sum(self.basis.get_scale_factors_for_k(student_dims))

        print(
            f"basis-bn (teacher_dim={teacher_dims}); scaling factor={self.scaling_factor} "
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=True, affine=True
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.scaling_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-bn-sum-normalized-learnable")
class OrthogonalBasisBatchNormSumNormalizedLearnablePolicy(
    OrthogonalBasisBatchNormSumNormalizedPolicy
):
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

        assert isinstance(self.transformer_teacher_feats, Adapter)

        self.transformer_teacher_feats.mat_encoder = nn.Parameter(
            self.transformer_teacher_feats.mat_encoder
        )

        if basis.centering:
            self.transformer_teacher_feats.mean = nn.Parameter(
                self.transformer_teacher_feats.mean
            )


@register_policy("basis-center")
class OrthogonalBasisCenterSumNormalizedPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = utils.modules.Centering2D(num_features=k).to(
            device
        )

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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


@register_policy("basis-center--teacher-feat-normalized")
class OrthogonalBasisCenterSumNormalizedPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.scaling_factor = (
            np.max(self.basis.get_scale_factors_for_k(student_dims)) ** 0.5
        )

        self.transformer_student_feats = utils.modules.Centering2D(num_features=k).to(
            device
        )

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats,
            transformed_teacher_feats / self.scaling_factor,
            reduction="none",
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


class CenteringWithScaling(nn.Module):
    # fixme: add test
    def __init__(self, k: int, init_scale: float):
        super().__init__()

        self.scale = torch.nn.Parameter(torch.tensor(init_scale))
        self.centering = utils.modules.Centering2D(num_features=k)

    def forward(self, feat: torch.Tensor):
        return self.scale * self.centering(feat)


@register_policy("basis-center--with-scaling")
class OrthogonalBasisCenterSumNormalizedPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = CenteringWithScaling(k=k, init_scale=1.0).to(
            device
        )

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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


class BasisOrthoStudenTransform(nn.Module):
    # fixme: add test
    def __init__(self, k: int, init_scale: float):
        super().__init__()

        self.scale = torch.nn.Parameter(torch.tensor(init_scale))
        self.rotation = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_features=k, out_features=k, bias=False)
        )

    def forward(self, feat: torch.Tensor):

        b, d, h, w = feat.shape

        feat = feat.reshape((b, d, h * w))

        feat = feat.permute(1, 0, 2)
        feat = feat.flatten(start_dim=1)
        # shape: [b*h*w, d]
        feat = feat.T
        feat = self.rotation(feat)

        # reshape back
        # shape: [d, b*h*w]
        feat = feat.T
        feat = feat.reshape(d, b, h * w)

        feat = feat.permute(1, 0, 2)
        feat = feat.reshape((b, d, h, w))

        return self.scale * feat


@register_policy("basis-center-rotation")
class OrthogonalBasisCenterRotationPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        init_scale = float(
            np.max(self.basis.get_scale_factors_for_k(student_dims)) ** 0.5
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k).to(device),
            BasisOrthoStudenTransform(k=k, init_scale=init_scale),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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


@register_policy("basis-center-rotation--scale-one-init")
class OrthogonalBasisCenterRotationPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        init_scale = 1.0

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k).to(device),
            BasisOrthoStudenTransform(k=k, init_scale=init_scale),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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


@register_policy("basis-center-rotation--scale-one-init-sum-normalized")
class OrthogonalBasisCenterRotationPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        if layerwise_training:
            # fixme: add tests
            self.scaling_factor = 1
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k).to(device),
            BasisOrthoStudenTransform(k=k, init_scale=1.0),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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


@register_policy("basis-center-rotation--scale-one-init-with-dimension-wise-weighting")
class OrthogonalBasisCenterRotationPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.weights = (
            torch.from_numpy(self.basis.get_scale_factors_for_k(k=k) ** 0.5)
            .float()
            .to(device)
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k).to(device),
            BasisOrthoStudenTransform(k=k, init_scale=1.0),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats,
            transformed_teacher_feats,
            reduction="none",
        ) / (w * h)

        # shape: (b, k)
        loss_mse = loss_mse.flatten(start_dim=2)
        assert loss_mse.shape == (b, k, w * h), loss_mse.shape

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=2)
        assert loss_mse.shape == (b, k), loss_mse.shape

        # weight each dimensions
        loss_mse = loss_mse * self.weights

        loss_mse = loss_mse.sum(dim=1)
        assert loss_mse.shape == (b,), loss_mse.shape

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


class BasisOrthoNoScaleStudenTransform(nn.Module):
    # fixme: add test
    def __init__(self, k: int):
        super().__init__()

        self.rotation = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_features=k, out_features=k, bias=False)
        )

    def forward(self, feat: torch.Tensor):

        b, d, h, w = feat.shape

        feat = feat.reshape((b, d, h * w))

        feat = feat.permute(1, 0, 2)
        feat = feat.flatten(start_dim=1)
        # shape: [b*h*w, d]
        feat = feat.T
        feat = self.rotation(feat)

        # reshape back
        # shape: [d, b*h*w]
        feat = feat.T
        feat = feat.reshape(d, b, h * w)

        feat = feat.permute(1, 0, 2)
        feat = feat.reshape((b, d, h, w))

        return feat


@register_policy("basis-center-rotation--no-scale")
class OrthogonalBasisCenterRotationPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k).to(device),
            BasisOrthoNoScaleStudenTransform(k=k),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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


@register_policy("basis-center-rotation--teacher-feat-and-scale-one-init")
class OrthogonalBasisCenterRotationTeacherFeatNormalizedPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.normalization_coeff = float(
            np.max(self.basis.get_scale_factors_for_k(student_dims)) ** 0.5
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k).to(device),
            BasisOrthoStudenTransform(k=k, init_scale=1.0),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats,
            transformed_teacher_feats / self.normalization_coeff,
            reduction="none",
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-rotation-center")
class OrthogonalBasisRotationCenterPolicy(LayerPolicy):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        init_scale = float(
            np.max(self.basis.get_scale_factors_for_k(student_dims)) ** 0.5
        )

        self.transformer_student_feats = nn.Sequential(
            BasisOrthoStudenTransform(k=k, init_scale=init_scale),
            utils.modules.Centering2D(num_features=k).to(device),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        init_scale = float(
            np.max(self.basis.get_scale_factors_for_k(student_dims)) ** 0.5
        )

        self.transformer_student_feats = nn.Sequential(
            BasisOrthoStudenTransform(k=k, init_scale=init_scale),
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
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
