import pytest
import numpy as np

from xaikd import bases


@pytest.mark.parametrize("centering", [False, True])
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
        expected_cov = (activation - mean).T @ (activation - mean) / n
    else:
        expected_cov = (activation).T @ (activation) / n

    suffix = "centered" if centering else "uncentered"

    pca = bases.get_basis(f"pca--{suffix}")

    eigvecs, eigvals = pca.fit(
        activation, None, mean=mean if centering else None, device="cpu"
    )

    np.testing.assert_allclose(eigvecs @ np.diag(eigvals) @ eigvecs.T, expected_cov)

    assert (eigvals[:-1] < eigvals[1:]).sum() == 0, "eigvalues are in descending order."

    if centering:
        assert f"{pca}" == "pca--centered"
    else:
        assert f"{pca}" == "pca--uncentered"


@pytest.mark.parametrize("centering", [False, True])
def test_prca(centering):
    np.random.seed(1)
    n, d = 10, 5

    if centering:
        # remark: here, mean is zero!
        activation = np.random.randn(n, d)
    else:
        # this adjustment makes sure that the mean is NOT zero
        activation = np.random.randn(n, d) + 2

    mean = activation.mean(axis=0)

    context = np.random.rand(n, d)

    if centering:
        centered_activation = activation - mean
        expected_cov = (
            (centered_activation.T @ context) + context.T @ centered_activation
        ) / n
    else:
        expected_cov = ((activation.T @ context) + context.T @ activation) / n

    suffix = "centered" if centering else "uncentered"

    prca = bases.get_basis(f"prca--{suffix}")

    eigvecs, eigvals = prca.fit(
        activation, context, mean=mean if centering else None, device="cpu"
    )

    np.testing.assert_allclose(eigvecs @ np.diag(eigvals) @ eigvecs.T, expected_cov)

    assert (
        eigvals[:-1] < eigvals[1:]
    ).sum() == 0, "Eigenvalues are in descending order"

    if centering:
        assert f"{prca}" == "prca--centered"
    else:
        assert f"{prca}" == "prca--uncentered"


@pytest.mark.parametrize("slug", ["centered", "uncentered"])
@pytest.mark.parametrize("seed", [1, 10])
def test_get_random_basis(slug, seed):
    basis = bases.get_basis(f"random{seed}--{slug}")

    assert hasattr(basis, "kwargs")
    assert basis.kwargs["seed"] == seed

    # todo: find way to test this  w/o relying on the mean artifacts.
    # basis.load(Path("dummy-path"))
