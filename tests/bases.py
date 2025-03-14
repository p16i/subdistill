import pytest
import numpy as np

import numpy.typing as npt


from xaikd import bases, utils


# fixme: add prcaposdefweighting
@pytest.mark.parametrize(
    "basis_name", ["pca", "gradpca", "prcasortabs", "prca", "prcaposdef"]
)
def test_analytic_basis(basis_name):
    rng = np.random.default_rng(seed=1)
    n, d, num_locations = 10, 4, 20

    arr_act = rng.random(size=(n, d, num_locations)) + 2
    arr_ctx = rng.random(size=(n, d, num_locations)) + 2
    arr_logodd = rng.random(size=(n,))
    logodd_threshold = 0.0

    mean = arr_act.mean(axis=0)

    arr_modified_act = arr_act

    basis = bases.get_basis(basis_name)

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        arr_logodd=arr_logodd,
        logodd_threshold=logodd_threshold,
    )

    arr_modified_act = utils.flatten_3d_tensor(arr_modified_act)
    arr_ctx = utils.flatten_3d_tensor(arr_ctx)
    arr_act = utils.flatten_3d_tensor(arr_act)

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
        basis.scale_factors, [np.mean((arr_modified_act @ basis.U[:, 0]) ** 2)]
    )


@pytest.mark.parametrize(
    "basis_name",
    [
        "pca",
        "gradpca",
        "prcasortabs",
        "prcaposdef",
        "prca-ablation-a-ac",
        "prca-ablation-c-ac",
        "prca-ablation-a-c",
    ],
)
def test_correct_scale_orthogoal_bases(basis_name):
    np.random.seed(1)
    n, d, num_locations = 10, 5, 20
    arr_act = np.random.randn(n, d, num_locations)
    arr_ctx = np.random.randn(n, d, num_locations)
    arr_logodd = np.random.randn(n)
    logodd_threshold = 0.0

    basis = bases.get_basis(basis_name)

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        arr_logodd=arr_logodd,
        logodd_threshold=logodd_threshold,
        seed=1,
    )

    U = basis.U
    scale = basis.scale_factors

    np.testing.assert_allclose(
        scale, [np.mean((utils.flatten_3d_tensor(arr_act) @ U[:, 0]) ** 2)]
    )


@pytest.mark.parametrize(
    "basis_name,mat_func,criteria",
    [
        ("pca", lambda d: d[0].T @ d[0], lambda x: x),
        ("pcacentering", lambda d: d[0].T @ d[0], lambda x: x),
    ],
)
def test_centering_orthogonal_bases(basis_name, mat_func, criteria):
    np.random.seed(1)
    n, d, num_locations = 10, 5, 20
    basis = bases.get_basis(basis_name)

    activation = np.random.randn(n, d, num_locations)
    context = np.random.randn(n, d, num_locations)
    arr_logodd = np.random.randn(
        n,
    )
    threshold = 0

    mean = np.mean(activation, axis=0)

    assert ("centering" in basis_name) == basis.centering

    modified_activation = activation - mean if basis.centering else activation

    expected_eigvals, expected_eigvecs = np.linalg.eigh(
        mat_func(
            (
                utils.flatten_3d_tensor(modified_activation),
                utils.flatten_3d_tensor(context),
            )
        )
    )

    expected_U = expected_eigvecs[:, np.argsort(-criteria(expected_eigvals))]

    basis.fit(
        arr_act=activation,
        arr_ctx=context,
        arr_logodd=arr_logodd,
        logodd_threshold=threshold,
        device="cpu",
    )

    actual_U = basis.U

    np.testing.assert_allclose(actual_U, expected_U)
