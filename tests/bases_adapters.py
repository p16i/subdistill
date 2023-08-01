import pytest
import numpy as np
from xaikd.utils import count_params_in_model

from xaikd import bases

from pathlib import Path
import tempfile


@pytest.mark.parametrize("basis_name", ["pca", "prca-recon", "prca-abs"])
def test_trainable_parameters_in_adapter(basis_name):
    seed = 1
    basis_name = f"{basis_name}--centered"

    basis = bases.get_basis(basis_name)

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
