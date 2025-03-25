import torch
from torch import nn
from torch.nn import functional as F

from ..register import register_policy
from ..interface import LastLayerPolicy

"""
@article{DBLP:journals/corr/HintonVD15,
    author       = {Geoffrey E. Hinton and
                    Oriol Vinyals and
                    Jeffrey Dean},
    title        = {Distilling the Knowledge in a Neural Network},
    journal      = {CoRR},
    volume       = {abs/1503.02531},
    year         = {2015}
}

The implementation is based on the following repositories:
    - https://github.com/yoshitomo-matsubara/torchdistill/blob/45ba679d4649512b520eb4ef7f97b757abf841ee/torchdistill/losses/mid_level.py#L100
    - https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/KD.py
"""


@register_policy("last-layer:kd")
class KLPolicy(LastLayerPolicy):

    # ref: temperature value from
    # https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L78
    def __init__(self, device: str, temperature=4):
        super().__init__()

        self.T = temperature
        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        kl = F.kl_div(
            torch.log_softmax(student_logits / self.T, dim=1),
            torch.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        )

        # We muliply with scaling factor as mentioned in Hinton et al. (2015)
        # (cf. the paragraph before Section 2.1).
        return (self.T**2) * kl


@register_policy("last-layer:binkd")
class BinaryKLPolicy(LastLayerPolicy):

    def __init__(self, device: str):
        super().__init__()

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        n = target.shape[0]

        assert teacher_logits.shape == student_logits.shape

        teacher_yp_gv_x = torch.sigmoid(teacher_logits)

        # fixme: check whether this is the special os KLDiv
        kl = F.binary_cross_entropy_with_logits(student_logits, teacher_yp_gv_x)

        return kl
