import pytest
import numpy as np
from xaikd.utils import count_params_in_model

from xaikd import bases, utils

import torch

from scipy.stats import ortho_group


def test_adapter():
    d = 20
    U = torch.from_numpy(ortho_group.rvs(d)).float()
    mean = torch.randn(d).float()
    encoder = bases.Adapter(
        U=U, mean=mean, mode=bases.AdapterMode.ENCODER, device="cpu"
    )
    decoder = bases.Adapter(
        U=U, mean=mean, mode=bases.AdapterMode.DECODER, device="cpu"
    )

    x = torch.randn(20, d, 1, 1)

    np.testing.assert_allclose(decoder(encoder(x)), x, atol=1e-5)


@pytest.mark.parametrize("basis_name", ["pca", "prcaposdef", "prcaposdef-entropy0.95"])
@pytest.mark.parametrize("d", [5, 10, 20])
def test_adapter_identity(basis_name, d):
    rng = np.random.default_rng(seed=1)
    arr_act = rng.random(size=(32, d, 10))
    mean = utils.flatten_3d_tensor(arr_act).mean(axis=0)
    arr_act -= mean[None, :, None]
    arr_ctx = rng.random(size=(32, d, 10))
    arr_logodd = rng.random(size=(32,))

    basis = bases.get_basis(f"{basis_name}")

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        mean_act=mean,
        arr_logodd=arr_logodd,
        logodd_threshold=0,
        device="cpu",
        strict_mode=True,
    )

    encoder = basis.construct_adapter(d, mode=bases.AdapterMode.ENCODER, device="cpu")
    decoder = basis.construct_adapter(d, mode=bases.AdapterMode.DECODER, device="cpu")

    x = torch.randn(20, d, 1, 1)

    np.testing.assert_allclose(decoder(encoder(x)), x, atol=1e-5)

    np.testing.assert_allclose(encoder.mean.numpy(), mean[None, :, None, None])


def test_adapter_zero_dim():
    d = 10
    k = 0

    basis_name = "pca"
    rng = np.random.default_rng(seed=1)
    arr_act = rng.random(size=(32, d, 10))
    mean = utils.flatten_3d_tensor(arr_act).mean(axis=0)
    arr_act -= mean[None, :, None]
    arr_ctx = rng.random(size=(32, d, 10))
    arr_logodd = rng.random(size=(32,))

    basis = bases.get_basis(f"{basis_name}")

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        mean_act=mean,
        arr_logodd=arr_logodd,
        logodd_threshold=0,
        device="cpu",
        strict_mode=True,
    )

    encoder = basis.construct_adapter(k, mode=bases.AdapterMode.ENCODER, device="cpu")
    decoder = basis.construct_adapter(k, mode=bases.AdapterMode.DECODER, device="cpu")

    x = torch.randn(20, d, 1, 1) * 0

    # we use the mean for the prediction
    expected = x * 0 + mean[None, :, None, None]

    np.testing.assert_allclose(decoder(encoder(x)), expected, atol=1e-5)


@pytest.mark.parametrize(
    "basis_name",
    [
        "pca",
        "prcasortabs",
        "prcaposdef",
        "prcaposdef-entropy0.95",
    ],
)
def test_no_trainable_parameters_in_adapter(basis_name):
    basis = bases.get_basis(basis_name)

    rng = np.random.default_rng(seed=1)

    act = rng.random(size=(47, 16, 10))
    mean_act = utils.flatten_3d_tensor(act).mean(axis=0)
    act -= mean_act[None, :, None]
    ctx = rng.random(size=(47, 16, 10))
    logodd = rng.random(size=(47,))

    basis.fit(
        arr_act=act,
        arr_ctx=ctx,
        mean_act=mean_act,
        arr_logodd=logodd,
        logodd_threshold=0,
        strict_mode=True,
    )

    for mode in [bases.AdapterMode.ENCODER, bases.AdapterMode.DECODER]:
        adapter = basis.construct_adapter(k=8, mode=mode, device="cpu")

        total, trainable = count_params_in_model(adapter)

        assert total == trainable == 0
