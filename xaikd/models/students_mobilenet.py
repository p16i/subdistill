from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

import torch
from torch import nn

from . import MODEL_GENERATORS


def _student_s_trained(num_classes, class_indices) -> nn.Module:

    assert class_indices is not None

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    fc: nn.Linear = model.classifier[3]

    fc.weight = nn.Parameter(fc.weight[class_indices, :])
    fc.bias = nn.Parameter(fc.bias[class_indices])

    return model


def _student_s(num_classes, **kwargs) -> nn.Module:
    return mobilenet_v3_small(num_classes=num_classes)


def _generate_model_function():

    MODEL_GENERATORS[f"student-mobilenets"] = _student_s
    MODEL_GENERATORS[f"student-mobilenets-trained"] = _student_s_trained


_generate_model_function()
