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

        if layerwise_training:
            self.scaling_factor = 1
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.RotateAndScale(k=k),
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
            self.scaling_factor = 1
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


class Bias(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias[None, :, None, None]


@register_policy("basis-center-rotation-with-bias")
class OrthogonalBasisCenterRotationWithBiasPolicy(LayerPolicy):
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
            self.scaling_factor = 1
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Rotate(k=k, bias=False), Bias(k=k)
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


@register_policy("basis-center-rotation-with-bias-only")
class OrthogonalBasisCenterRotationWithBiasPolicy(LayerPolicy):
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
            self.scaling_factor = 1
        else:
            self.scaling_factor = np.sum(
                self.basis.get_scale_factors_for_k(student_dims)
            )

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device, use_mean=True
        )

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k, bias=False),
            Bias(k=k),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        transformed_teacher_feats = (
            transformed_teacher_feats / (self.scaling_factor) ** 0.5
        )

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


####


class AblationTemplate(LayerPolicy):

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

        self.transformer_teacher_feats = self._construct_teacher_transformation(
            basis=basis, k=k, device=device
        )

        self.transformer_student_feats = self._construct_student_transformation(
            k=k, device=device
        )

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

    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        raise NotImplementedError()

    def _construct_student_transformation(self, k: int, device: str):
        raise NotImplementedError()


class Normalization(nn.Module):
    def __init__(self, normalization_constant: float):
        super().__init__()

        self.constant = normalization_constant

    def forward(self, x):
        return x / self.constant


@register_policy(
    "basis-ablationv2--l2--teacher-center-normalized--student-center-rotation"
)
class AblationNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        scaling_factor = float(basis.get_scale_factors_for_k(k=k).sum() ** 0.5)
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            Normalization(scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy("basis-ablationv2--l2--teacher-center--student-center-rotation")
class AblationNoNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy("basis-ablationv2--l2--teacher-center--student-center")
class AblationNoNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
        ).to(device)


@register_policy("basis-ablationv2--l2normalized--teacher-center--student-center")
class AblationNoNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
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

        loss_mse = loss_mse / np.sum(self.basis.get_scale_factors_for_k(k))

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy(
    "basis-ablationv2--l2normalized--teacher-center--student-center-linear"
)
class AblationNormalizedTeacherCenterLinear(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):

        (d, _) = basis.U.shape
        # here, we don't perform any projection
        return nn.Sequential(
            basis.construct_adapter(k=d, mode=AdapterMode.ENCODER, device=device),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        (d, _) = self.basis.U.shape
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            nn.Conv2d(in_channels=k, out_channels=d, bias=False, kernel_size=1),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape
        (d, _) = self.basis.U.shape

        loss_scale_factor = np.sum(self.basis.get_scale_factors_for_k(d))

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats,
            transformed_teacher_feats,
            reduction="none",
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / loss_scale_factor

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy(
    "basis-ablationv2--l2normalized--teacher-center--student-center-linearortho"
)
class AblationNormalizedTeacherCenterLinearOrtho(AblationNormalizedTeacherCenterLinear):

    def _construct_student_transformation(self, k: int, device: str):
        (d, _) = self.basis.U.shape
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.LinearOrtho(in_features=k, out_features=d),
        ).to(device)


@register_policy("basis-ablationv2--l2--teacher-center-normalized--student-center")
class AblationNormalizedTeacherCenter(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        scaling_factor = float(basis.get_scale_factors_for_k(k=k).sum() ** 0.5)
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            Normalization(scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
        ).to(device)


@register_policy("basis-ablationv2--l2--teacher-projection--student-rotation")
class AblationNoNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):

        U = torch.from_numpy(basis.U[:, :k]).float()
        # here, we don't subtract mean.
        mean = torch.zeros_like(torch.from_numpy(basis.mean))

        return nn.Sequential(
            Adapter(U=U, mean=mean, mode=AdapterMode.ENCODER, device=device)
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy(
    "basis-ablationv2--l2normalized--teacher-projection--student-rotation-scale"
)
class AblationNoNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):

        U = torch.from_numpy(basis.U[:, :k]).float()
        # here, we don't subtract mean.
        mean = torch.zeros_like(torch.from_numpy(basis.mean))

        return nn.Sequential(
            Adapter(U=U, mean=mean, mode=AdapterMode.ENCODER, device=device)
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.RotateAndScale(k=k),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        normalization_coeff = np.sum(self.basis.get_scale_factors_for_k(k))

        loss_mse = F.mse_loss(
            transformed_student_feats,
            transformed_teacher_feats,
            reduction="none",
        ) / (w * h)
        loss_mse = loss_mse / normalization_coeff

        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy(
    "basis-ablationv2--l2--teacher-projection-normalized--student-identity"
)
class AblationNoNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):

        scaling_factor = float(basis.get_scale_factors_for_k(k=k).sum() ** 0.5)

        U = torch.from_numpy(basis.U[:, :k]).float()
        # here, we don't subtract mean.
        mean = torch.zeros_like(torch.from_numpy(basis.mean))

        return nn.Sequential(
            Adapter(U=U, mean=mean, mode=AdapterMode.ENCODER, device=device),
            Normalization(scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Identity()


@register_policy("basis-ablationv2--lcos--teacher-center--student-center-rotation")
class AblationLossCosNoNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        transformed_teacher_feats = F.normalize(transformed_teacher_feats, dim=1)
        transformed_student_feats = F.normalize(transformed_student_feats, dim=1)

        cosine = (transformed_teacher_feats * transformed_student_feats).sum(dim=1)
        cosine = cosine / (w * h)

        loss = -cosine

        loss = loss.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss = loss.sum(dim=1)

        assert loss.shape == (b,)

        # average over all samples
        loss = loss.mean()

        return loss


@register_policy("basis-ablationv2--l2--teacher-center--student-center-linear")
class AblationIdentityTeacherCenterLinear(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        _, d = basis.U.shape
        return nn.Sequential(
            # we perform only centering and no projection
            basis.construct_adapter(k=d, mode=AdapterMode.ENCODER, device=device),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):

        _, d = self.basis.U.shape
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            nn.Conv2d(in_channels=k, out_channels=d, bias=False, kernel_size=1),
        ).to(device)


@register_policy("basis-ablationv2--l2--teacher-center--student-center-linearortho")
class AblationIdentityTeacherCenterLinearOrtho(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        _, d = basis.U.shape
        return nn.Sequential(
            # we perform only centering and no projection
            basis.construct_adapter(k=d, mode=AdapterMode.ENCODER, device=device),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):

        _, d = self.basis.U.shape
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False),
            utils.modules.LinearOrtho(in_features=k, out_features=d),
        ).to(device)


@register_policy(
    "basis-ablationv2--l2--teacher-center-normalized--student-center-linearortho"
)
class AblationNormalizedTeacherCenterLinearOrtho(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        _, d = basis.U.shape
        scaling_factor = float(basis.get_scale_factors_for_k(k=k).sum() ** 0.5)

        return nn.Sequential(
            # we perform only centering and no projection
            basis.construct_adapter(k=d, mode=AdapterMode.ENCODER, device=device),
            Normalization(scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):

        _, d = self.basis.U.shape
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False),
            utils.modules.LinearOrtho(in_features=k, out_features=d),
        ).to(device)


@register_policy(
    "basis-ablationv2--l2--teacher-center-normalized--student-center-linear"
)
class AblationIdentityTeacherCenterLinear(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        _, d = basis.U.shape

        scaling_factor = float(basis.get_scale_factors_for_k(k=k).sum() ** 0.5)
        return nn.Sequential(
            # we perform only centering and no projection
            basis.construct_adapter(k=d, mode=AdapterMode.ENCODER, device=device),
            Normalization(scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):

        _, d = self.basis.U.shape
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            nn.Conv2d(in_channels=k, out_channels=d, bias=False, kernel_size=1),
        ).to(device)


#####
