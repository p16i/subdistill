import numpy as np

import torch
from torch import nn

from ..register import register_policy
from ..interface import LayerPolicy

"""
@inproceedings{DBLP:conf/cvpr/AhnHDLD19,
    author       = {Sungsoo Ahn and
                    Shell Xu Hu and
                    Andreas C. Damianou and
                    Neil D. Lawrence and
                    Zhenwen Dai},
    title        = {Variational Information Distillation for Knowledge Transfer},
    booktitle    = {{CVPR}},
    pages        = {9163--9171},
    publisher    = {Computer Vision Foundation / {IEEE}},
    year         = {2019}
}
Remark: The code below is based on the implementation in https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/VID.py#L9C1-L54C20
"""


@register_policy("vid")
class VIDPolicy(LayerPolicy):

    # See Supplement  (P4330), the line before the last paragraph.
    init_pred_var = 5.0
    eps = 1e-5

    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        # cf. https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/VID.py#L20
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                stride=stride,
            )

        # cf.
        # - https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L223
        # - https://github.com/yoshitomo-matsubara/torchdistill/blob/e615746172f653ebb59a29b396effba5aa766942/configs/sample/ilsvrc2012/vid/resnet18_from_resnet50.yaml#L123
        num_input_channels = student_dims
        num_mid_channel = num_target_channels = teacher_dims

        # ref: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py#L26
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        ).to(device)

        # ref: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py#L33
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(self.init_pred_var - self.eps) - 1.0)
            * torch.ones(num_target_channels)
        ).to(device)

        self.eps = self.eps

        self.transformer_student_feats = self.regressor
        self.transformer_teacher_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        target = transformed_teacher_feats
        pred_mean = transformed_student_feats

        # cf. the Paragraph after eq. (5)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)

        # cf. eq. (6)
        # ref: https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py#L50
        # remark: pred_var = std^2. Therefore, we have torch.log(pred_var) = 2 * log(std).
        # But, in eq 6, we have log(std) instead of 2 * log(std), hence having 0.5 * torch.log(pred_var).
        neg_log_prob = 0.5 * (
            ((pred_mean - target) ** 2) / pred_var + torch.log(pred_var)
        )

        # cf. https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py#L53
        # cf. https://github.com/yoshitomo-matsubara/torchdistill/blob/74a710e882a85204ca27233477695d08086ca7b1/torchdistill/losses/mid_level.py#L696
        loss = torch.mean(neg_log_prob)

        return loss
