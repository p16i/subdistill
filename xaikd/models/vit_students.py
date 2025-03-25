import typing

from collections import OrderedDict

import torch
from torch import nn
import numpy as np

from functools import partial

from torchvision.models.vision_transformer import _vision_transformer

from xaikd import constants

from . import add_model_to_registry
from . import vit


def _generate_vit_student(num_layers: int, num_classes: int, hidden_dim: int, **kwargs):
    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L621
    model = _vision_transformer(
        patch_size=16,
        num_layers=num_layers,
        num_heads=12,
        hidden_dim=hidden_dim,
        mlp_dim=3072 // 4,
        weights=None,
        progress=False,
        num_classes=num_classes,
    )

    vit.make_encoder_intermediate_output_have_cnn_like_shape_(model)

    return model


def _generate_model_function():

    for hidden_dim in constants.ARR_VIT_STUDENT_HIDDEN_DIMENSIONS:

        add_model_to_registry(
            f"vitstudent-{hidden_dim}",
            partial(_generate_vit_student, hidden_dim=hidden_dim, num_layers=4),
        )

        add_model_to_registry(
            f"vitstudent6l-{hidden_dim}",
            partial(_generate_vit_student, hidden_dim=hidden_dim, num_layers=6),
        )


_generate_model_function()
