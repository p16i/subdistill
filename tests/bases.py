import pytest
import numpy as np

import numpy.typing as npt


from xaikd import bases


def is_permuation_matrix(x: npt.NDArray) -> bool:
    # ref: https://stackoverflow.com/a/28896366
    return (
        x.ndim == 2
        and x.shape[0] == x.shape[1]
        and (x.sum(axis=0) == 1).all()
        and (x.sum(axis=1) == 1).all()
        and ((x == 1) | (x == 0)).all()
    )


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


@pytest.mark.parametrize("centering", [True])
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

    suffix = "centered" if centering else "uncentered"

    prca = bases.get_basis(f"prca--{suffix}")

    eigvecs, std = prca.fit(
        activation, context, mean=mean if centering else None, device="cpu"
    )

    np.testing.assert_allclose(std, np.std(centered_activation @ eigvecs, axis=0))

    if centering:
        assert f"{prca}" == "prca--centered"
    else:
        assert f"{prca}" == "prca--uncentered"


@pytest.mark.parametrize("basis_name", ["random", "randomperm"])
@pytest.mark.parametrize("slug", ["centered"])
@pytest.mark.parametrize("seed", [1, 10])
def test_get_random_basis(basis_name, slug, seed):
    basis = bases.get_basis(f"{basis_name}--{slug}", seed=seed)

    assert hasattr(basis, "kwargs")
    assert basis.kwargs["seed"] == seed

    if basis_name == "randomperm":
        activation = np.random.randn(20, 5)
        context = np.random.randn(20, 5)
        mean = np.mean(activation, axis=0)
        U, _ = basis.fit(activation, context, mean=mean, device="cpu")

        assert is_permuation_matrix(U)


@pytest.mark.parametrize("variant", ["abs", "recon", "reconreg0.1"])
@pytest.mark.parametrize("slug", ["centered"])
def test_instantiate_prca_greedy_basese(variant, slug):
    basis = bases.get_basis(f"prca-{variant}--{slug}")
    assert True


@pytest.mark.parametrize(
    "basis_name", ["pca", "rel", "random", "prca-abs", "prca-recon"]
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


@pytest.mark.parametrize("basis_name", ["rel", "rel-abs", "act", "act-abs"])
def test_correctness_canonical_based_manual_case_pos(basis_name):
    mode = "uncentered"

    activation = np.array([[1, 3, 2, 5]])
    _, d = activation.shape
    context = activation

    mean = activation.mean(axis=0)

    basis = bases.get_basis(f"{basis_name}--{mode}", seed=1)

    eigvecs, std = basis.fit(
        activation=activation, context=context, mean=mean, device="cpu"
    )

    assert is_permuation_matrix(eigvecs)

    np.testing.assert_allclose(eigvecs, np.eye(d)[:, [3, 1, 2, 0]])


def test_correctness_canonical_based_manual_case_neg():
    mode = "uncentered"

    activation = np.array([[1, -3, 2, 5]])
    _, d = activation.shape
    context = np.array([[1, -3, 2, -5]])

    mean = activation.mean(axis=0)

    for basis_name, expected_order in [
        ("rel-abs", [3, 1, 2, 0]),
        ("rel", [1, 2, 0, 3]),
        ("act-abs", [3, 1, 2, 0]),
        ("act", [3, 2, 0, 1]),
    ]:
        basis = bases.get_basis(f"{basis_name}--{mode}", seed=1)
        eigvecs, _ = basis.fit(
            activation=activation, context=context, mean=mean, device="cpu"
        )
        np.testing.assert_allclose(eigvecs, np.eye(d)[:, expected_order])


def test_correctness_canonical_based_manual_case_pos_mag_rel_reverse():
    mode = "uncentered"

    activation = np.array([[1, 3, 2, 5]])
    _, d = activation.shape
    context = np.array([[1, 1e-3, 1e-2, 1e-5]])

    mean = activation.mean(axis=0)

    for basis_name, expected_order in [
        ("rel-abs", [0, 2, 1, 3]),
        ("rel", [0, 2, 1, 3]),
        ("act-abs", [3, 1, 2, 0]),
        ("act", [3, 1, 2, 0]),
    ]:
        basis = bases.get_basis(f"{basis_name}--{mode}", seed=1)
        eigvecs, _ = basis.fit(
            activation=activation, context=context, mean=mean, device="cpu"
        )
        np.testing.assert_allclose(eigvecs, np.eye(d)[:, expected_order])


@pytest.mark.parametrize("basis_name", ["rel", "rel-abs"])
def test_correctness_rel_random_cases(basis_name):
    basis_name = "rel"
    mode = "uncentered"

    np.random.seed(1)
    n, d = 20, 5
    activation = np.random.randn(n, d)
    context = activation

    mean = activation.mean(axis=0)

    basis = bases.get_basis(f"{basis_name}--{mode}", seed=1)

    eigvecs, std = basis.fit(
        activation=activation, context=context, mean=mean, device="cpu"
    )

    expected_order = np.argsort(np.mean(activation**2, axis=0))[::-1]

    assert is_permuation_matrix(eigvecs)

    np.testing.assert_allclose(eigvecs, np.eye(d)[:, expected_order])
