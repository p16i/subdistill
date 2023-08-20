import pytest
import numpy as np
from xaikd.utils import count_params_in_model

from xaikd import bases

import torch

from scipy.stats import ortho_group
import tempfile

from pathlib import Path


def test_adapter():
    d = 20
    U = torch.from_numpy(ortho_group.rvs(d)).float()
    mean = torch.randn(d).float()
    std = torch.rand(d).float()
    encoder = bases.Adapter(
        U=U, mean=mean, std=std, mode=bases.AdapterMode.ENCODER, device="cpu"
    )
    decoder = bases.Adapter(
        U=U, mean=mean, std=std, mode=bases.AdapterMode.DECODER, device="cpu"
    )

    x = torch.randn(20, d, 1, 1)

    np.testing.assert_allclose(decoder(encoder(x)), x, atol=1e-5)


@pytest.mark.parametrize("mode", ["centered", "uncentered"])
@pytest.mark.parametrize("k", [5, 10, 20])
def test_adapter_identity(mode, k):
    d = 20

    np.random.seed(1)
    arr_act = np.random.rand(32, d)
    mean = np.mean(arr_act, axis=0)

    basis = bases.get_basis(f"identity--{mode}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        np.save(tmpdirname / "act_mean", mean)
        basis.fit(arr_act, None, mean=mean, device="cpu")
        basis.save(tmpdirname)
        basis.load(tmpdirname)

        encoder = basis.construct_adapter(
            k, mode=bases.AdapterMode.ENCODER, device="cpu"
        )
        decoder = basis.construct_adapter(
            k, mode=bases.AdapterMode.DECODER, device="cpu"
        )

        x = torch.randn(20, d, 1, 1)

        np.testing.assert_allclose(decoder(encoder(x)), x, atol=1e-5)


@pytest.mark.parametrize("basis_name", ["pca", "prca-recon", "prca-abs"])
def test_trainable_parameters_in_adapter(basis_name):
    seed = 1
    basis_name = f"{basis_name}--centered"

    basis = bases.get_basis(basis_name, seed=1)

    np.random.seed(1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        act = np.random.randn(47, 16)
        mean = np.mean(act, axis=0)
        ctx = np.random.randn(47, 16)

        np.save(tmpdirname / "act_mean.npy", mean)

        basis.fit(act, ctx, mean=mean, device="cpu")
        basis.save(tmpdirname)
        basis.load(tmpdirname)

        for mode in [bases.AdapterMode.ENCODER, bases.AdapterMode.DECODER]:
            adapter = basis.construct_adapter(k=8, mode=mode, device="cpu")

            total, trainable = count_params_in_model(adapter)

            assert total == trainable == 0
