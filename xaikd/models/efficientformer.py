import pytest

import timm
from torch import nn

from . import add_model_to_registry


def _construct_effcientformerv2s0(num_classes: int, **kwargs) -> nn.Module:
    return timm.create_model("efficientformerv2_s0", num_classes=num_classes)


def _add_model_to_register():
    add_model_to_registry("student-efficientformerv2_s0", _construct_effcientformerv2s0)


_add_model_to_register()
