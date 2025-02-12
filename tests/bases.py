import pytest
import numpy as np

import numpy.typing as npt


from xaikd import bases


def test_solve_eigh():
    rng = np.random.default_rng(seed=1)
    arr_act = rng.integers(
        0,
        10,
        size=(10, 5),
    )
    arr_ctx = rng.integers(0, 10, size=(10, 5))

    # case 1: psd
    cov = arr_act.T @ arr_act

    actual_cov_eigvals, actual_cov_eigvecs = bases._solve_eigh(cov=cov)
    expected_cov_eigvals, expected_cov_eigvecs = np.linalg.eigh(cov)

    np.testing.assert_allclose(
        actual_cov_eigvals,
        np.flip(expected_cov_eigvals),
    )

    assert (actual_cov_eigvals >= 0).all()

    np.testing.assert_allclose(
        actual_cov_eigvecs,
        np.flip(expected_cov_eigvecs, axis=1),
    )

    # case 2: indefinite
    ccov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

    # case 2.1: sort raw eigvals
    ccov_eigvals, ccov_eigvecs = np.linalg.eigh(ccov)
    assert (np.mean(ccov_eigvals >= 0) > 0) and (np.mean(ccov_eigvals < 0) > 0)

    actual_ccov_eigvals, actual_ccov_eigvecs = bases._solve_eigh(cov=ccov)
    np.testing.assert_allclose(actual_ccov_eigvals, np.flip(ccov_eigvals))
    np.testing.assert_allclose(actual_ccov_eigvecs, np.flip(ccov_eigvecs, axis=1))

    # case 2.2: sort abs eigvals
    actual_ccov_abs_eigvals, actual_ccov_abs_eigvecs = bases._solve_eigh(
        cov=ccov, sort_with_abs_eigvals=True
    )
    ccov_abs_eigvals = np.abs(ccov_eigvals)
    sorted_abs_idx = np.argsort(-ccov_abs_eigvals)

    expected_ccov_abs_eigvals = ccov_abs_eigvals[sorted_abs_idx]
    expected_ccov_abs_eigvecs = ccov_eigvecs[:, sorted_abs_idx]

    np.testing.assert_allclose(actual_ccov_abs_eigvals, expected_ccov_abs_eigvals)
    np.testing.assert_allclose(actual_ccov_abs_eigvecs, expected_ccov_abs_eigvecs)


@pytest.mark.parametrize(
    "basis_name", ["pca", "gradpca", "prcasortabs", "prca", "prcaposdef"]
)
def test_analytic_basis(basis_name):
    rng = np.random.default_rng(seed=1)
    n, d = 10, 4

    arr_act = rng.random(size=(n, d)) + 2
    arr_ctx = rng.random(size=(n, d)) + 2

    mean = arr_act.mean(axis=0)

    arr_modified_act = arr_act

    basis: bases.Orthogonal = bases.get_basis(basis_name)

    basis.fit(arr_act, arr_ctx)

    # compute expected U
    expected_U = None
    if basis_name == "pca":
        expected_U = np.flip(
            np.linalg.eigh(arr_modified_act.T @ arr_modified_act)[1], axis=1
        )
    elif basis_name == "gradpca":
        expected_U = np.flip(np.linalg.eigh(arr_ctx.T @ arr_ctx)[1], axis=1)
    elif basis_name == "prcaposdef":

        cov_a = arr_act.T @ arr_act
        cov_c = arr_ctx.T @ arr_ctx

        cov_ac = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        cov_posdef = 2 * (
            cov_a / np.trace(cov_a) + cov_c / np.trace(cov_c)
        ) + cov_ac / np.power(np.trace(cov_a) * np.trace(cov_c), 0.5)

        expected_U = np.flip(np.linalg.eigh(cov_posdef)[1], axis=1)

    elif basis_name in ["prca", "prcasortabs"]:
        eigvals, eigvecs = np.linalg.eigh(
            arr_modified_act.T @ arr_ctx + arr_ctx.T @ arr_modified_act
        )
        print(f"Ttest: eigvals: {eigvals}")

        if basis_name == "prca":
            expected_U = np.flip(eigvecs, axis=1)
        elif basis_name == "prcasortabs":
            expected_U = eigvecs[:, np.argsort(-np.abs(eigvals))]

    if expected_U is None:
        raise ValueError(f"{basis_name} has no expected_U!")
    # end

    # verification
    if basis.centering:
        np.testing.assert_allclose(basis.mean, mean)
    else:
        np.testing.assert_allclose(basis.mean, np.zeros(d))

    np.testing.assert_allclose(basis.U, expected_U)

    np.testing.assert_allclose(
        basis.scale_factors, np.mean((arr_modified_act @ basis.U) ** 2, axis=0)
    )


@pytest.mark.parametrize(
    "basis_name",
    [
        "random",
        "pca",
        "gradpca",
        "prcasortabs",
        "prcaposdef",
    ],
)
def test_correct_scale_orthogoal_bases(basis_name):
    np.random.seed(1)
    n, d = 10, 5
    arr_act = np.random.randn(n, d)
    arr_ctx = np.random.randn(n, d)

    basis: bases.Orthogonal = bases.get_basis(basis_name)

    basis.fit(arr_act=arr_act, arr_ctx=arr_ctx, seed=1)

    U = basis.U
    scale = basis.scale_factors

    np.testing.assert_allclose(scale, np.mean((arr_act @ U) ** 2, axis=0))


@pytest.mark.parametrize(
    "basis_name,mat_func,criteria",
    [
        ("pca", lambda d: d[0].T @ d[0], lambda x: x),
        ("pcacentering", lambda d: d[0].T @ d[0], lambda x: x),
    ],
)
def test_centering_orthogonal_bases(basis_name, mat_func, criteria):
    np.random.seed(1)
    n, d = 10, 5
    basis: bases.Orthogonal = bases.get_basis(basis_name)

    activation = np.random.randn(n, d)
    context = np.random.randn(n, d)

    mean = np.mean(activation, axis=0)

    assert ("centering" in basis_name) == basis.centering

    modified_activation = activation - mean if basis.centering else activation

    expected_eigvals, expected_eigvecs = np.linalg.eigh(
        mat_func((modified_activation, context))
    )

    expected_U = expected_eigvecs[:, np.argsort(-criteria(expected_eigvals))]

    basis.fit(arr_act=activation, arr_ctx=context, device="cpu")

    actual_U = basis.U

    np.testing.assert_allclose(actual_U, expected_U)
