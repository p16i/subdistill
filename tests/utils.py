import pytest

from torch import nn

import numpy as np

from xaikd import utils


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
