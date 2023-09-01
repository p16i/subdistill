import torch

from torch import nn
from torch.nn import functional as F

from xaikd.bases import Basis, AdapterMode


class KL(nn.Module):
    # refs:
    # - https://github.com/yoshitomo-matsubara/torchdistill/blob/45ba679d4649512b520eb4ef7f97b757abf841ee/torchdistill/losses/mid_level.py#L100
    # - https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/KD.py

    def __init__(self, temperature: float):
        super().__init__()

        self.T = temperature

    def forward(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor
    ) -> torch.Tensor:
        assert len(teacher_logits.shape) == 2
        assert teacher_logits.shape == student_logits.shape

        kl = F.kl_div(
            torch.log_softmax(student_logits / self.T, dim=1),
            torch.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        )

        # remark: we muliply with scaling factor
        # as mentioned in Hinton et al. (2015) (paragraph before Section 2.1)
        return (self.T**2) * kl


class BasisL2Loss(nn.Module):
    def __init__(self, basis: Basis, k: int, device: str) -> None:
        super().__init__()

        self.basis = basis
        self.transform_teacher = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

    def forward(self, teacher_feats: torch.Tensor, student_feats) -> torch.Tensor:
        transformed_teacher_feats = self.transform_teacher(teacher_feats)

        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == student_feats.shape

        loss_mse = F.mse_loss(
            student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse
