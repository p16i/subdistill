import torch
from torch import nn
from torch.nn import functional as F

from ..register import register_policy
from ..interface import LayerPolicy

"""
@InProceedings{Miles_2024_CVPR,
    author    = {Miles, Roy and Elezi, Ismail and Deng, Jiankang},
    title     = {VkD: Improving Knowledge Distillation using Orthogonal Projections},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {15720-15730}
}
"""


class StudentTransformer(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()

        self.orthogonal_transform = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_features=in_dims, out_features=out_dims, bias=False)
        )

    def forward(self, feat: torch.Tensor):
        b, d, h, w = feat.shape

        # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L94
        # remark: there, the mean of the student  feature is over sequence of tokens
        # which is last dimensions in our case
        # and it is similar to https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L96
        feat = feat.reshape((b, d, h * w))

        feat = feat.mean(-1)

        return self.orthogonal_transform(feat)


class TeacherTransformer(nn.Module):
    def forward(self, feat: torch.Tensor):
        b, d, h, w = feat.shape

        # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L96
        feat = feat.reshape((b, d, h * w))
        feat = feat.mean(-1)

        return F.layer_norm(feat, normalized_shape=(d,))


@register_policy("vkd")
class VkDPolicy(LayerPolicy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__()

        self.transformer_student_feats = StudentTransformer(
            in_dims=student_dims, out_dims=teacher_dims
        )
        self.transformer_teacher_feats = TeacherTransformer()

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:

        (
            b,
            _,
        ) = transformed_student_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # remark: they use smooth_l1_loss(...) but why?
        # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L100
        loss_mse = (transformed_student_feats - transformed_teacher_feats) ** 2
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (
            b,
        ), f"shape: {loss_mse.shape}, student: {transformed_student_feats.shape}; teacher: {transformed_teacher_feats.shape}"

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse
