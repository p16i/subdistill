import typing

from collections import OrderedDict

import torch
from torch import nn
import numpy as np

from functools import partial

from torchvision.models.vision_transformer import _vision_transformer

from xaikd import constants

from . import MODEL_GENERATORS
from . import vit


def _generate_vit_student(num_classes: int, hidden_dim: int, **kwargs):
    # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L621
    model = _vision_transformer(
        patch_size=16,
        num_layers=4,
        num_heads=12,
        hidden_dim=hidden_dim,
        mlp_dim=3072 // 4,
        weights=None,
        progress=False,
        num_classes=num_classes,
    )

    vit.make_encoder_intermediate_output_have_cnn_like_shape_(model)

    # ?todo: add adapter module in each of these block

    return model


def _generate_model_function():

    for hidden_dim in constants.ARR_VIT_STUDENT_HIDDEN_DIMENSIONS:

        MODEL_GENERATORS[f"vitstudent-{hidden_dim}"] = partial(
            _generate_vit_student, hidden_dim=hidden_dim
        )


_generate_model_function()
