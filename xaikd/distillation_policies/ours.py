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

        if layerwise_training:
            self.scaling_factor = 1.0
        else:
            self.scaling_factor = float(
                np.sum(self.basis.get_scale_factors_for_k(student_dims))
            )

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


@register_policy("basis-ablation--normalized-teacher--center-rotation")
class AblationNormalizedTeacherCenterRotation(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            Normalization(self.scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Rotate(k=k),
        ).to(device)


@register_policy("basis-ablation--no-normalized-teacher--center-rotation")
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


@register_policy("basis-ablation--center-teacher--center-linear")
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


@register_policy("basis-ablation--center-normalized-teacher--center-linear")
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
            Normalization(float(np.sum(basis.get_scale_factors_for_k(d)))),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):

        _, d = self.basis.U.shape
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            nn.Conv2d(in_channels=k, out_channels=d, bias=False, kernel_size=1),
        ).to(device)


@register_policy("basis-ablation--normalized-teacher--center-rotation-scale")
class AblationNormalizedTeacherCenterRotationScale(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            Normalization(self.scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.RotateAndScale(k=k),
        ).to(device)


@register_policy("basis-ablation--normalized-teacher--scale-bias")
class AblationNormalizedTeacherCenterRotationScaleBias(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            Normalization(self.scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Scale(),
            utils.modules.Bias(k=k),
        ).to(device)


@register_policy(
    "basis-ablation--normalized-teacher--center-rotation--no-spatial-normalization"
)
class AblationNormalizedTeacherCenterRotationNoSpatialNormalization(
    AblationNormalizedTeacherCenterRotation
):
    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats,
            transformed_teacher_feats,
            reduction="none",
        )
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-ablation--normalized-teacher--center")
class AblationNormalizedTeacherCenter(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            Normalization(self.scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
        ).to(device)


@register_policy("basis-ablation--normalized-teacher--center-scale")
class AblationNormalizedTeacherCenter(AblationTemplate):
    def _construct_teacher_transformation(
        self,
        basis: OrthogonalBasis,
        k: int,
        device: str,
    ):
        return nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            Normalization(self.scaling_factor),
        ).to(device)

    def _construct_student_transformation(self, k: int, device: str):
        return nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False).to(device),
            utils.modules.Scale(),
        ).to(device)


#######


@register_policy("basis-center-rotation-normalized-teacher")
class OrthogonalBasisCenterRotationNormalizedTeacherPolicy(LayerPolicy):
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

        transformed_teacher_feats = transformed_teacher_feats / self.scaling_factor

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


@register_policy("basis-center-rotation-normalized-teacher-no-scale")
class OrthogonalBasisCenterRotationNormalizedTeacherPolicy(LayerPolicy):
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

        transformed_teacher_feats = transformed_teacher_feats / self.scaling_factor

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


@register_policy("basis-bias-rotation-normalized-teacher-no-scale")
class OrthogonalBasisCenterRotationNormalizedTeacherPolicy(LayerPolicy):
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
            utils.modules.Bias(k=k),
            utils.modules.Rotate(k=k),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        transformed_teacher_feats = transformed_teacher_feats / self.scaling_factor

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


@register_policy("basis-rotation-normalized-teacher-no-scale")
class OrthogonalBasisCenterRotationNormalizedTeacherPolicy(LayerPolicy):
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
            utils.modules.Rotate(k=k),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        transformed_teacher_feats = transformed_teacher_feats / self.scaling_factor

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


@register_policy("basis-center-rotation-no-normalization-and-scale")
class OrthogonalBasisCenterRotationNormalizedTeacherPolicy(LayerPolicy):
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

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-rotation-with-bias-normalized-teacher")
class OrthogonalBasisCenterRotationNormalizedTeacherPolicy(LayerPolicy):
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
            utils.modules.RotateWithBiasAndScale(k=k),
        ).to(device)

    def criterion(
        self, transformed_teacher_feats, transformed_student_feats
    ) -> torch.Tensor:
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        transformed_teacher_feats = transformed_teacher_feats / self.scaling_factor

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
