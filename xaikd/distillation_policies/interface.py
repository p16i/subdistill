import typing

from abc import ABC, abstractmethod

import torch
from torch import nn
from torch.nn import functional as F


class Policy(nn.Module, ABC):
    transformer_teacher_feats: nn.Module
    transformer_student_feats: nn.Module

    @abstractmethod
    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def align_spatial_dimensions(
        self, teacher_feats, student_feats
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        pass

    def forward(self, teacher_feats, student_feats) -> torch.Tensor:
        teacher_feats, student_feats = self.align_spatial_dimensions(
            teacher_feats, student_feats
        )

        transformed_teacher_feats = self.transformer_teacher_feats(teacher_feats)
        transformed_student_feats = self.transformer_student_feats(student_feats)

        assert (
            transformed_student_feats.shape == transformed_teacher_feats.shape
        ), f"{transformed_student_feats.shape}; {transformed_teacher_feats.shape}"

        return self.criterion(transformed_teacher_feats, transformed_student_feats)


class LastLayerPolicy(Policy, ABC):

    def forward(  # type: ignore
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.criterion(
            teacher_logits=teacher_logits, student_logits=student_logits, target=target
        )

    @abstractmethod
    def criterion(  # type: ignore
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def align_spatial_dimensions(
        self, teacher_feats, student_feats
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return teacher_feats, student_feats


class LayerPolicy(Policy):
    def align_spatial_dimensions(self, teacher_feats, student_feats):

        _, _, teacher_height, teacher_width = teacher_feats.shape
        _, _, student_height, student_width = student_feats.shape

        if (student_height == teacher_height) and (student_width == teacher_width == 1):
            # for ViT
            pass
        else:
            # for CNN
            assert teacher_height == teacher_width
            assert student_height == student_width

            if student_height != teacher_height:
                # cf. https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/FT.py#L18C7-L24C17
                # we assume that the feture maps are square.
                assert teacher_height == teacher_width
                assert student_height == student_width

                if student_height > teacher_height:
                    student_feats = F.adaptive_avg_pool2d(
                        student_feats, (teacher_height, teacher_height)
                    )
                elif student_height < teacher_height:
                    teacher_feats = F.adaptive_avg_pool2d(
                        teacher_feats, (student_height, student_height)
                    )

        return teacher_feats, student_feats


class LayerPolicyCollection(nn.ModuleList):
    def __init__(
        self,
        teacher_layers: typing.List[str],
        student_layers: typing.List[str],
        policies: typing.List[LayerPolicy],
    ) -> None:
        super().__init__(policies)

        assert len(teacher_layers) == len(student_layers) == len(policies)

        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.policies = policies
