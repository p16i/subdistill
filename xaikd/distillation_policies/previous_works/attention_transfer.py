import torch
from torch import nn
from torch.nn import functional as F

from ..interface import LayerPolicy
from ..register import register_policy

"""
@inproceedings{DBLP:conf/iclr/ZagoruykoK17,
    author       = {Sergey Zagoruyko and
                    Nikos Komodakis},
    title        = {Paying More Attention to Attention: Improving the Performance of Convolutional
                    Neural Networks via Attention Transfer},
    booktitle    = {{ICLR} (Poster)},
    publisher    = {OpenReview.net},
    year         = {2017}
}
"""


class AttentionMappingFsumP2(nn.Module):
    def __init__(self, ord: int):
        super().__init__()
        self.ord = ord

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.shape

        # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L19C4-L19C61
        # also Section 3.1 in the paper
        return F.normalize(
            x.pow(self.ord).mean(1).view(x.size(0), -1), dim=1, p=self.ord
        )


@register_policy("attention-transfer")
class AttentionTransferPolicy(LayerPolicy):

    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, ord=2
    ) -> None:
        super().__init__()

        self.ord = ord

        self.transformer_student_feats = AttentionMappingFsumP2(ord=self.ord)
        self.transformer_teacher_feats = AttentionMappingFsumP2(ord=self.ord)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):

        # comparing spatial dimensions
        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L22
        # Remark: the implementation seems to be different from eq (2) in the paper.
        # In particular, the difference is that calling `.mean()` has a factor of 1/(wh),
        # while the original equal (eq.2) has a factor of `1`.
        loss_mse = (
            (transformed_teacher_feats - transformed_student_feats).pow(self.ord).mean()
        )

        return loss_mse
