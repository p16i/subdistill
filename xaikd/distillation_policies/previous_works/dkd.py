import torch
from torch import nn
from torch.nn import functional as F

from ..interface import LastLayerPolicy
from ..register import register_policy

"""
@inproceedings{DBLP:conf/cvpr/ZhaoCSQL21,
    author       = {Borui Zhao and
                    Quan Cui and
                    Renjie Song and
                    Yiyu Qiu and
                    Jiajun Liang},
    title        = {Decoupled Knowledge Distillation},
    booktitle    = {{CVPR}},
    pages        = {11942--11952},
    publisher    = {{IEEE}},
    year         = {2021}
}
"""


""
# The followings are taken from https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


###


@register_policy("last-layer:dkd")
class DKDPolicy(LastLayerPolicy):

    def __init__(
        self,
        device: str,
        # cf. https://github.com/megvii-research/mdistiller/blob/master/configs/imagenet/r34_r18/dkd.yaml#L29
        temperature=1.0,
        # cf. Section 4.1
        alpha=1.0,
        beta=8.0,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return dkd_loss(
            logits_student=student_logits,
            logits_teacher=teacher_logits,
            target=target,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        )
