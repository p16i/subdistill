import pytest
import numpy as np
from xaikd.utils import count_params_in_model

from xaikd import bases

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


@pytest.mark.parametrize("basis_name", ["pca", "prcasortabs", "prcaposdef"])
@pytest.mark.parametrize("d", [5, 10, 20])
def test_adapter_identity(basis_name, d):

    rng = np.random.default_rng(seed=1)
    arr_act = rng.random(size=(32, d))
    arr_ctx = rng.random(size=(32, d))

    basis = bases.get_basis(basis_name)

    basis.fit(arr_act, arr_ctx, device="cpu")

    encoder = basis.construct_adapter(d, mode=bases.AdapterMode.ENCODER, device="cpu")
    decoder = basis.construct_adapter(d, mode=bases.AdapterMode.DECODER, device="cpu")

    x = torch.randn(20, d, 1, 1)

    np.testing.assert_allclose(decoder(encoder(x)), x, atol=1e-5)


@pytest.mark.parametrize(
    "basis_name", ["pca", "pcacentering", "prcasortabs", "prcaposdef"]
)
def test_trainable_parameters_in_adapter(basis_name):

    basis = bases.get_basis(basis_name)

    rng = np.random.default_rng(seed=1)

    act = rng.random(size=(47, 16))
    ctx = rng.random(size=(47, 16))

    basis.fit(act, ctx)

    for mode in [bases.AdapterMode.ENCODER, bases.AdapterMode.DECODER]:
        adapter = basis.construct_adapter(k=8, mode=mode, device="cpu")

        total, trainable = count_params_in_model(adapter)

        assert total == trainable == 0
