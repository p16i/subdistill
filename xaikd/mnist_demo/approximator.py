import torch

import numpy as np
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from xaikd import bases


class Approximator(nn.Module):
    def __init__(self, k, kernel_size: int, input_channels=1):
        super().__init__()

        self.k = k
        kernel_size = kernel_size

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=k, kernel_size=kernel_size, padding="valid"
        )

        self.conv2 = nn.Conv2d(
            in_channels=k,
            out_channels=k,
            kernel_size=(1, 1),
            padding="valid",
            bias=True,
        )

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)

        return out

    def __str__(self):
        return f"Approx(k={self.k},w={self.conv1.weight.shape})"


class ApproximatorModelWrapper(pl.LightningModule):
    def __init__(
        self,
        approx,
        teacher,
        basis: bases.Basis,
        k: int,
        lambda_mse: float,
        lambda_xent: float,
        verbose=False,
        device="cpu",
    ):
        super().__init__()

        self.approx = approx
        self.teacher = teacher

        self.verbose = verbose

        self.lambda_mse = lambda_mse
        self.lambda_xent = lambda_xent
        self.encoder = basis.construct_projection_on_rank_k(k, device=device)
        self.decoder = basis.contruct_rank_d_decoder(k, device=device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.approx.parameters(), lr=1e-3)
        return optimizer

    def compute_loss(self, batch, prefix):
        x, y = batch

        with torch.no_grad():
            act = self.teacher.forward_feat(x)
            projected_act = self.encoder(act)

            b, c, h, w = projected_act.shape

        spatial_scaling = 1 / (h * w)
        approxed_projected_act = self.approx(x)

        # compute MSE loss
        mse = (
            F.mse_loss(approxed_projected_act, projected_act, reduction="none")
            * spatial_scaling
        )

        mse = mse.flatten(start_dim=1)
        mse = mse.sum(dim=1)

        # compute crossent loss
        decoded_approxed_projected_act = self.decoder(approxed_projected_act).flatten(
            start_dim=1
        )
        logits = self.teacher.lin2(decoded_approxed_projected_act)

        loss_mse = self.lambda_mse * mse.mean()
        loss_xent = self.lambda_xent * F.cross_entropy(logits, y)

        loss = loss_mse + loss_xent

        self.log(f"{prefix}_loss_xent", loss_xent, on_epoch=True)
        self.log(f"{prefix}_loss_mse", loss_mse, on_epoch=True)
        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "val")


class CombinedModule(nn.Module):
    def __init__(
        self,
        approximator: Approximator,
        teacher: nn.Module,
        basis: bases.Basis,
        device="cpu",
    ):
        super().__init__()

        self.approximator = approximator
        self.teacher = teacher

        self.decoder = basis.contruct_rank_d_decoder(approximator.k, device=device)

    def forward(self, x):
        latent = self.approximator(x)

        approxed_act = self.decoder(latent)

        approxed_act = approxed_act.flatten(start_dim=1)

        x = self.teacher.lin2(approxed_act)

        return x
