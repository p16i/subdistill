import pytest

import torch
import numpy as np

from xaikd.bases import learners
from xaikd import bases


@pytest.mark.parametrize("name", ["abs", "recon", "reconnaive", "reconreg0.1"])
def test_obj_first_dimensions(name):
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

    elif name in ["recon", "reconnaive"]:
        mode, beta = name, 0.0
        expected = -((rel_projected - rel_original) ** 2)
    elif "reconreg" in name:
        mode = "recon"
        beta = float(name.replace("reconreg", ""))
        expected = -((rel_projected - rel_original) ** 2)
        expected += beta * (act_projected).abs()

    learner = learners.PRCAGreedyLeaner(mode=mode)
    IUUt = torch.eye(2)
    actual = learner.obj_func(act, ctx, IUUt, u, beta)

    assert actual == expected.mean()


@pytest.mark.parametrize(
    "name",
    ["abs", "recon", "reconnaive", "reconreg0.1"],
)
def test_obj_secon_dimensions(name):
    act = torch.tensor([[5, 1], [-2, 3]]).float()
    ctx = torch.tensor([[1, -1], [1, 1]]).float()

    u1 = torch.tensor([1, 0]).float()

    IUUt = torch.eye(2) - u1.outer(u1)

    u2 = torch.tensor([0, 1]).float()

    rel_ori = ((act) * (ctx)).sum(axis=1)
    rel_comp = ((act @ IUUt) * (ctx @ IUUt)).sum(axis=1)
    act_projected = act @ u2
    ctx_projected = ctx @ u2
    rel_projected = act_projected * ctx_projected

    if name == "abs":
        mode, beta = name, 0.0
        expected = torch.abs(rel_projected)
    elif name == "recon":
        mode, beta = name, 0.0
        expected = -((rel_projected - rel_comp) ** 2)
    elif name == "reconnaive":
        mode, beta = name, 0.0
        expected = -((rel_projected - rel_ori) ** 2)
    elif "reconreg" in name:
        mode = "recon"
        beta = float(name.replace("reconreg", ""))
        expected = -((rel_projected - rel_comp) ** 2)
        expected += beta * (act_projected).abs()
    else:
        raise

    learner = learners.PRCAGreedyLeaner(mode=mode)
    actual = learner.obj_func(act, ctx, IUUt, u2, beta)

    assert actual == expected.mean()


@pytest.mark.parametrize("mode", ["abs", "recon", "reconnaive"])
def test_learner_trainable(mode):
    np.random.seed(1)

    act = np.random.randn(10, 2)
    ctx = np.random.randn(10, 2)

    learner = learners.PRCAGreedyLeaner(mode=mode)

    U1 = learner.fit(act, ctx, epochs=2, seed=1)

    U2 = learner.fit(act, ctx, epochs=2, seed=1)

    np.testing.assert_allclose(U1, U2)


@pytest.mark.parametrize("mode", ["centered"])
@pytest.mark.parametrize(
    "basis_name",
    [
        "prca-abs",
        "prca-recon",
        "prca-reconnaive",
        "pcaprca-abs",
        "pcaprca-recon",
    ],
)
def test_calling_learner_from_basis(basis_name, mode):
    np.random.seed(1)

    act = np.random.randn(10, 2)
    ctx = np.random.randn(10, 2)

    mean = act.mean(axis=0)

    prca1 = bases.get_basis(f"{basis_name}--{mode}", seed=1)
    U1, std1 = prca1.fit(act, ctx, mean=mean, device="cpu")

    np.testing.assert_allclose(std1, np.std(act @ U1, axis=0))

    prca2 = bases.get_basis(f"{basis_name}--{mode}", seed=1)
    U2, std2 = prca2.fit(act, ctx, mean=mean, device="cpu")

    np.testing.assert_allclose(U1, U2)
    np.testing.assert_allclose(std1, std2)
