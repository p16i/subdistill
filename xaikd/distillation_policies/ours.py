import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from xaikd.bases import OrthogonalBasis, AdapterMode
from xaikd import utils

from .register import register_policy
from .interface import LayerPolicy


@register_policy("nothing")
class NothingPolicy(LayerPolicy):
    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        return torch.tensor(0.0).to(transformed_teacher_feats.device)


@register_policy("basis-with-bias-and-scale")
class OrthogonalBasisIdentityBiasAndScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class AddBiasAndScale(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(d).reshape(1, d, 1, 1))
                self.scaling = nn.Parameter(torch.tensor(1.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.scaling * x + self.bias

        self.transformer_student_feats = AddBiasAndScale(d=k).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

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


@register_policy("basis-with-bias")
class OrthogonalBasisIdentityBiasPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class AddBias(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(d).reshape(1, d, 1, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.bias

        self.transformer_student_feats = AddBias(d=k).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

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


@register_policy("basis-with-biasv2")
class OrthogonalBasisIdentityBiasV2Policy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class AddBias(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(d).reshape(1, d, 1, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.bias

        self.transformer_student_feats = AddBias(d=k).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k=k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-with-biasv3")
class OrthogonalBasisIdentityBiasV2Policy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class AddBias(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(d).reshape(1, d, 1, 1))
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return F.tanh(self.scale) * x + self.bias

        self.transformer_student_feats = AddBias(d=k).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k=k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-bn-only-affine")
class OrthogonalBasisIdentityBatchNormOnlyAffinePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=False, affine=True
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-centering")
class OrthogonalBasisCenteringPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        assert self.basis.centering
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        print("Scaling factor:", self.basis.get_scale_factors_for_k(k).max())

        self.transformer_student_feats = utils.modules.Centering2d(
            num_features=k,
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-centering-and-scale")
class OrthogonalBasisCenteringAndScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis

        assert self.basis.centering

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class StudentTransformer(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.centering = utils.modules.Centering2d(
                    num_features=k,
                )
                self.scale = nn.Parameter(torch.tensor(1.0))

            def forward(self, x: torch.Tensor):
                return self.scale * self.centering(x)

        print("Scaling factor:", self.basis.get_scale_factors_for_k(k).max())

        self.transformer_student_feats = StudentTransformer(d=k).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-centering-and-scale-constant")
class OrthogonalBasisCenteringAndScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis

        assert self.basis.centering

        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class StudentTransformer(nn.Module):
            def __init__(self, d: int, scale: float):
                super().__init__()
                self.centering = utils.modules.Centering2d(
                    num_features=k,
                )
                self.scale = scale

            def forward(self, x: torch.Tensor):
                return self.scale * self.centering(x)

        self.transformer_student_feats = StudentTransformer(
            d=k,
            scale=self.basis.get_scale_factors_for_k(k).max() ** 0.5,
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

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


@register_policy("basis-centering-and-scale-constant-teacher")
class OrthogonalBasisCenteringAndScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis

        assert self.basis.centering

        class TeacherTransformer(nn.Module):
            def __init__(self, encoder: nn.Module, scale: float):
                super().__init__()
                self.encoder = encoder
                self.scale = scale

            def forward(self, x: torch.Tensor):
                return self.encoder(x) / self.scale

        self.transformer_teacher_feats = TeacherTransformer(
            encoder=basis.construct_adapter(
                k=k, mode=AdapterMode.ENCODER, device=device
            ),
            scale=self.basis.get_scale_factors_for_k(k).max() ** 0.5,
        ).to(device)

        self.transformer_student_feats = utils.modules.Centering2D(num_features=k).to(
            device
        )

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

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


@register_policy("basis-bn-only-runstats")
class OrthogonalBasisIdentityBatchNormOnlyRunStatsPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=True, affine=False
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-bn")
class OrthogonalBasisIdentityBatchNormPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        scaling_factors = self.basis.get_scale_factors_for_k(student_dims)
        sqrt_tr = np.sum(scaling_factors) ** 0.5
        print(
            f"basis-bn (teacher_dim={teacher_dims}); scaling factor: max={scaling_factors.max():.4e}, first={scaling_factors[0]:4e}, sqrt(sum_k lambda_k)={sqrt_tr:.4e}"
        )
        self.scaling = np.max(scaling_factors)

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

        loss_mse = loss_mse / self.scaling

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-bn-no-scale")
class OrthogonalBasisIdentityBatchNormNoScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
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

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-bn-only-mean-no-scale")
class OrthogonalBasisIdentityBatchNormOnlyMeanNoScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = utils.modules.BatchNormOnlyMean(
            num_features=k, track_running_stats=True, affine=False
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

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


@register_policy("basis-bn-only-mean")
class OrthogonalBasisIdentityBatchNormOnlyMeanPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = utils.modules.BatchNormOnlyMean(
            num_features=k, track_running_stats=True, affine=False
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-rotation")
class OrthogonalBasisRotationPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        class StudenTransform(nn.Module):
            def __init__(self):
                super().__init__()

                self.rotation = nn.utils.parametrizations.orthogonal(
                    nn.Linear(
                        in_features=student_dims, out_features=student_dims, bias=False
                    )
                )

            def forward(self, x):
                x = utils.convolve_feature_map_with_linear(x, self.rotation)

                return x

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = StudenTransform()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_policy("basis-rotation-with-scale")
class OrthogonalBasisRotationWithScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:

        super().__init__()

        k = student_dims

        class StudenTransform(nn.Module):
            def __init__(self):
                super().__init__()

                self.rotation = nn.utils.parametrizations.orthogonal(
                    nn.Linear(
                        in_features=student_dims, out_features=student_dims, bias=False
                    )
                )

                self.scaling = nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                x = utils.convolve_feature_map_with_linear(x, self.rotation)

                return x * self.scaling

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = StudenTransform()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse
