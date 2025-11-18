import numpy as np
from numpy import typing as npt

from pytorch_lightning import LightningModule
import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.nn import functional as F

from xaikd.bases import OrthogonalBasis
from xaikd import bases, logit_modifiers

from .register import register_policy
from .interface import LayerPolicy, PolicyWithLogging, PolicyWithFitSteps


from xaikd import utils


class SubtractMean(nn.Module):
    def __init__(self, mean: npt.NDArray, device) -> None:
        super().__init__()

        self.mean = torch.from_numpy(mean).reshape((1, -1, 1, 1)).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.mean


@register_policy("convergence-linear")
class OrthogonalPCAConvergenceWithLinearPolicy(
    LayerPolicy, PolicyWithLogging, PolicyWithFitSteps
):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__()

        k = student_dims
        d = teacher_dims

        self.d = d
        self.k = k

        self.transformer_student_feats = nn.Conv2d(
            in_channels=k,
            out_channels=d,
            bias=False,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def fit_for_teacher(
        self,
        teacher: nn.Module,
        teacher_layer: str,
        teacher_dims: int,
        student_dims: int,
        train_loader: DataLoader,
        seed: int,
        device: str,
    ) -> None:
        pass

        self.dict_Uk = {}

        logit_mod = logit_modifiers.MultiClassDifferenceTop2Logits()

        print("Fitting bases...")

        for bix, basis_name in enumerate(["pca", "pcarev", "random"]):
            basis = bases.helpers.learn_basis(
                teacher_model=teacher,
                train_loader=train_loader,
                logit_mod=logit_mod,
                layer=teacher_layer,
                basis_name=basis_name,
                device=device,
                seed=seed,
            )

            Uk = torch.from_numpy(basis.get_Uk(student_dims)).float().to(device)

            if bix == 0:
                self.transformer_teacher_feats = SubtractMean(basis.mean, device)

            self.dict_Uk[basis_name] = Uk

    def fit_for_student(
        self,
        student: nn.Module,
        student_layer: str,
        teacher_dims: int,
        student_dims: int,
        train_loader: DataLoader,
        seed: int,
        device: str,
    ) -> None:
        pass

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

        for key, Uk in self.dict_Uk.items():
            Uk = self.dict_Uk[key]

            recon = F.conv2d(
                self.transformer_student_feats(student_feat),
                (Uk @ Uk.T).unsqueeze(2).unsqueeze(3),
            )

            err = ref - recon

            err_norm = torch.linalg.norm(err, dim=1) ** 2
            recon_norm = torch.linalg.norm(recon, dim=1) ** 2
            ref_norm = torch.linalg.norm(ref, dim=1) ** 2

            np.testing.assert_allclose(
                (err_norm + recon_norm).detach().cpu().numpy(),
                ref_norm.detach().cpu().numpy(),
                atol=1e-5,
            )

            relative_recon = ((recon_norm / (ref_norm + 1e-8))).mean()

            module.log(
                f"{prefix}_recon_ration_on_basis_{key}", relative_recon, on_epoch=True
            )


@register_policy("convergence-nocenter-linear")
class OrthogonalPCAConvergenceNoCenterWithLinearOrthoPolicy(
    OrthogonalPCAConvergenceWithLinearPolicy
):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device)

        k = student_dims
        d = teacher_dims

        self.d = d
        self.k = k

        self.transformer_teacher_feats = nn.Identity()

        self.transformer_student_feats = nn.Sequential(
            nn.Conv2d(
                in_channels=k,
                out_channels=d,
                bias=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        ).to(device)


@register_policy("convergence-nocenter-linear-ortho")
class OrthogonalPCAConvergenceNoCenterWithLinearPolicy(
    OrthogonalPCAConvergenceWithLinearPolicy
):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device)

        k = student_dims
        d = teacher_dims

        self.d = d
        self.k = k

        self.transformer_teacher_feats = nn.Identity()

        self.transformer_student_feats = nn.Sequential(
            utils.modules.LinearOrtho(
                in_features=k,
                out_features=d,
                bias=False,
            )
        ).to(device)


@register_policy("convergence-center-linear")
class OrthogonalPCAConvergenceWithCenterLinearPolicy(
    OrthogonalPCAConvergenceWithLinearPolicy
):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device)

        k = student_dims
        d = teacher_dims

        self.d = d
        self.k = k

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False),
            nn.Conv2d(
                in_channels=k,
                out_channels=d,
                bias=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        ).to(device)


@register_policy("convergence-linear-ortho")
class OrthogonalPCAConvergenceWithLinearOrthoPolicy(
    OrthogonalPCAConvergenceWithLinearPolicy
):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device)

        k = student_dims
        d = teacher_dims

        self.d = d
        self.k = k

        self.transformer_student_feats = utils.modules.LinearOrtho(
            in_features=k,
            out_features=d,
            bias=False,
        ).to(device)


@register_policy("convergence-center-linear-ortho")
class OrthogonalPCAConvergenceWithCenterLinearOrthoPolicy(
    OrthogonalPCAConvergenceWithLinearPolicy
):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__(teacher_dims, student_dims, device)

        k = student_dims
        d = teacher_dims

        self.d = d
        self.k = k

        self.transformer_student_feats = nn.Sequential(
            utils.modules.Centering2D(num_features=k, affine=False),
            utils.modules.LinearOrtho(in_features=k, out_features=d, bias=False),
        ).to(device)
