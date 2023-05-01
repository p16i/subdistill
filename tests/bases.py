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

    expected_mean = activation.mean(axis=0)
    if centering:
        expected_cov = (activation - expected_mean).T @ (activation - expected_mean) / n
    else:
        expected_cov = (activation).T @ (activation) / n

    pca = bases.get_basis("pca", centering=centering)

    eigvecs, mean, eigvals = pca.fit(activation, None)

    if centering:
        np.testing.assert_allclose(mean, expected_mean)
    else:
        np.testing.assert_allclose(mean, np.zeros(d))

    np.testing.assert_allclose(eigvecs @ np.diag(eigvals) @ eigvecs.T, expected_cov)

    assert (eigvals[:-1] < eigvals[1:]).sum() == 0

    if centering:
        assert f"{pca}" == "pca-centered"
    else:
        assert f"{pca}" == "pca-uncentered"


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

    expected_mean = activation.mean(axis=0)

    context = np.random.rand(n, d)

    if centering:
        centered_activation = activation - expected_mean
        expected_cov = (
            (centered_activation.T @ context) + context.T @ centered_activation
        ) / n
    else:
        expected_cov = ((activation.T @ context) + context.T @ activation) / n

    prca = bases.get_basis("prca", centering=centering)

    eigvecs, mean, eigvals = prca.fit(activation, context)

    if centering:
        np.testing.assert_allclose(mean, expected_mean)
    else:
        np.testing.assert_allclose(mean, np.zeros(d))

    np.testing.assert_allclose(eigvecs @ np.diag(eigvals) @ eigvecs.T, expected_cov)

    assert (eigvals[:-1] < eigvals[1:]).sum() == 0

    if centering:
        assert f"{prca}" == "prca-centered"
    else:
        assert f"{prca}" == "prca-uncentered"
