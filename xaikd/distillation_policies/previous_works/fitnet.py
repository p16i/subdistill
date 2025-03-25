import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from ..register import register_policy
from ..interface import LayerPolicy

"""
@inproceedings{DBLP:journals/corr/RomeroBKCGB14,
    author       = {Adriana Romero and
                    Nicolas Ballas and
                    Samira Ebrahimi Kahou and
                    Antoine Chassang and
                    Carlo Gatta and
                    Yoshua Bengio},
    title        = {FitNets: Hints for Thin Deep Nets},
    booktitle    = {{ICLR} (Poster)},
    year         = {2015}
}
Remark: the implementation is the FitNet version in VID setup (cf. Ahn et al. (2019, VID, First Paragraph of Section 3).
The difference from the original setup is that, here, we do NOT pretrain the linear transform of the student.
Said differently, the FitNet implementation here contains only ONE stage.
"""


@register_policy("fitnet-relu")
class FitNet(LayerPolicy):

    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_student_feats = self._build_student_feats_transfomer(
            teacher_dims=teacher_dims, student_dims=student_dims
        ).to(device)

        self.transformer_teacher_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

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

    def _build_student_feats_transfomer(
        self, teacher_dims: int, student_dims: int
    ) -> nn.Module:
        return nn.Sequential(
            # cf.
            # 1. https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/models/util.py#L131
            # 2. https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/adaptation.py#L33
            nn.Conv2d(
                out_channels=teacher_dims,
                in_channels=student_dims,
                # cf. https://github.com/yoshitomo-matsubara/torchdistill/blob/ba62248b48cefeea24f1ef774a870f338711a9d9/configs/sample/ilsvrc2012/fitnet/resnet18_from_resnet152.yaml#L119
                kernel_size=1,
            ),
            nn.ReLU(),
        )


@register_policy("fitnet-noact")
class FitNetTwoLayers(FitNet):
    def _build_student_feats_transfomer(
        self, teacher_dims: int, student_dims: int
    ) -> nn.Module:
        return nn.Conv2d(
            out_channels=teacher_dims,
            in_channels=student_dims,
            kernel_size=1,
        )
