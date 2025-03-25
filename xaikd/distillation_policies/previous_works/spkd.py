import torch
from torch import nn
from torch.nn import functional as F

from ..register import register_policy
from ..interface import LayerPolicy

# Tung & Mori 2019, Similarity-Preserving Knowledge Distillation


class SimilarityMatrixConstructor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.shape

        # eq 2
        x = x.reshape((b, c * w * h))
        x = F.normalize(x, dim=-1)

        return x @ x.T


@register_policy("spkd")
class SimilarityPreserveKnowledgeDistillationPolicy(LayerPolicy):

    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, ord=2
    ) -> None:
        super().__init__()

        self.ord = ord

        self.transformer_student_feats = SimilarityMatrixConstructor()
        self.transformer_teacher_feats = SimilarityMatrixConstructor()

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:

        b, _ = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # eq 4
        score = torch.norm(
            transformed_teacher_feats - transformed_student_feats, p="fro"
        ) / (b**2)
        return score

    def align_spatial_dimensions(self, teacher_feats, student_feats):
        # remark: we don't do any transformation here.
        return teacher_feats, student_feats
