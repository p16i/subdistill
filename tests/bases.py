import pytest
import numpy as np

import numpy.typing as npt


from xaikd import bases


@pytest.mark.parametrize("centering", [True])
def test_pca(centering):
    np.random.seed(1)
    n, d = 10, 5

    if centering:
        activation = np.random.randn(n, d)
    else:
        # this adjustment makes sure that the mean is NOT zero
        activation = np.random.randn(n, d) + 2

    mean = activation.mean(axis=0)

    if centering:
        centered_activation = activation - mean
        expected_cov = centered_activation.T @ centered_activation / n
    else:
        expected_cov = (activation).T @ (activation) / n

    suffix = "centered" if centering else "uncentered"

    pca = bases.get_basis(f"pca--{suffix}")

    eigvecs, std = pca.fit(
        activation, None, mean=mean if centering else None, device="cpu"
    )

    eigvals = std**2

    np.testing.assert_allclose(eigvecs @ np.diag(eigvals) @ eigvecs.T, expected_cov)

    np.testing.assert_allclose(std, np.std(centered_activation @ eigvecs, axis=0))

    assert (eigvals[:-1] < eigvals[1:]).sum() == 0, "eigvalues are in descending order."

    if centering:
        assert f"{pca}" == "pca--centered"
    else:
        assert f"{pca}" == "pca--uncentered"


@pytest.mark.parametrize("basis_mode", ["centered", "uncentered"])
@pytest.mark.parametrize(
    "basis_name,criteria",
    [
        ("prca", lambda eigvals: eigvals),
        ("prca-sortabs", lambda eigvals: np.abs(eigvals)),
    ],
)
def test_prca(basis_name, basis_mode, criteria):
    np.random.seed(1)
    n, d = 10, 5

    activation = np.random.randn(n, d) + 2
    context = np.random.rand(n, d)

    if basis_mode == "centered":
        mean = activation.mean(axis=0)
    else:
        mean = np.zeros(d)

    centered_activation = activation - mean

    crosscov = ((centered_activation.T @ context) + context.T @ centered_activation) / n

    eigvals, eigvecs = np.linalg.eigh(crosscov)

    basis = bases.get_basis(f"{basis_name}--{basis_mode}")

    U, std = basis.fit(activation, context, mean=mean, device="cpu")

    expected_U = eigvecs[:, np.argsort(-criteria(eigvals))]

    np.testing.assert_allclose(U, expected_U)

    np.testing.assert_allclose(std, np.std(centered_activation @ U, axis=0))

    assert f"{basis}" == f"{basis_name}--{basis_mode}"


@pytest.mark.parametrize("variant", ["abs", "recon", "reconreg0.1"])
@pytest.mark.parametrize("slug", ["centered"])
def test_instantiate_prca_greedy_basese(variant, slug):
    basis = bases.get_basis(f"prca-{variant}--{slug}")
    assert True


@pytest.mark.parametrize(
    "basis_name", ["pca", "rel-raw", "random", "prca-abs", "prca-recon"]
)
def test_correct_std(basis_name):
    mode = "centered"

    np.random.seed(1)
    n, d = 10, 5
    activation = np.random.randn(n, d)
    context = np.random.randn(n, d)

    mean = activation.mean(axis=0)

    centered_activation = activation - mean

    basis = bases.get_basis(f"{basis_name}--{mode}", seed=1)

    eigvecs, std = basis.fit(
        activation=activation, context=context, mean=mean, device="cpu"
    )

    np.testing.assert_allclose(std, np.std(centered_activation @ eigvecs, axis=0))
