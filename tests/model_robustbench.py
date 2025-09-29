import pytest

import torch

from xaikd import models


def test():
    model = models.get_trained_model("cifar100-modas2021-robustbench")

    x = torch.randn(2, 3, 32, 32)

    y = model(x)
