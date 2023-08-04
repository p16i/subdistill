import pytest

from torch import nn

import numpy as np

from xaikd import utils
from xaikd import models


def test_subsample():
    np.random.seed(1)
    act = np.random.randn(10, 3, 5, 5)
    ctx = np.random.randn(10, 3, 5, 5)
    subsampled_act, subsampled_ctx = utils.subsample_tensors(act, ctx, num_locations=13)

    assert subsampled_act.shape == subsampled_ctx.shape
    assert subsampled_act.shape == (10 * 13, 3)


def test_count_params():
    lin1 = nn.Linear(20, 16)

    lin2 = nn.Linear(16, 7)

    utils.deactivate_requires_grad(lin2)

    model = nn.Sequential(lin1, lin2)

    total, trainable = utils.count_params_in_model(model)

    assert total == ((20 + 1) * 16 + (16 + 1) * 7)
    assert trainable == ((20 + 1) * 16)


def test_query_module_with_types():
    module = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=1),
        nn.BatchNorm2d(32),
        nn.Sequential(nn.Conv2d(32, 64, kernel_size=1), nn.BatchNorm2d(64)),
        nn.Conv2d(1, 64, kernel_size=1),
        nn.BatchNorm2d(64),
        nn.Linear(64, 10),
    )

    arr_batchnorm = utils.query_module_children_with_type(module, nn.BatchNorm2d)
    arr_conv2d = utils.query_module_children_with_type(module, nn.Conv2d)
    arr_linear = utils.query_module_children_with_type(module, nn.Linear)

    assert len(arr_batchnorm) == 3
    assert len(arr_conv2d) == 3
    assert len(arr_linear) == 1


def test_query_module_with_types_resnet18_cifar100():
    module = models.get_model("cifar100-resnet18-p1")

    arr_batchnorm = utils.query_module_children_with_type(module, nn.BatchNorm2d)

    assert len(arr_batchnorm) == 20
