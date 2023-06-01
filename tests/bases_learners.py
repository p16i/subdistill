import pytest

import torch
import numpy as np

from xaikd.bases import learners


@pytest.mark.parametrize("name", ["abs", "recon", "reconreg0.1"])
def test_obj(name):
    act = torch.tensor([[5, 1], [-2, 3]]).float()
    ctx = torch.tensor([[1, -1], [1, 1]]).float()

    u = torch.tensor([1, 0]).float()

    act_projected = act @ u
    ctx_projected = ctx @ u

    rel_original = (act * ctx).sum(axis=1)
    rel_projected = act_projected * ctx_projected


    if name == "abs":
        mode, beta = name, 0.0
        expected = torch.abs(rel_projected)

    elif name == "recon":
        mode, beta = name, 0.0
        expected = -((rel_projected - rel_original) ** 2)
    elif "reconreg" in name:
        mode = "recon"
        beta = float(name.replace("reconreg", ""))
        expected = -((rel_projected - rel_original) ** 2)
        expected += beta * (act_projected).abs()

    learner = learners.PRCAGreedyLeaner(mode=mode)

    actual = learner.obj_func(act, ctx, u, beta)

    assert actual == expected.mean()


@pytest.mark.parametrize("mode", ["abs", "recon"])
def test_learner_trainable(mode):
    np.random.seed(1)

    act = np.random.randn(10, 2)
    ctx = np.random.randn(10, 2)

    learner = learners.PRCAGreedyLeaner(mode=mode)

    learner.fit(act, ctx, epochs=2)

    assert True
