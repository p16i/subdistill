from numpy._typing import NDArray
import pytest
import typing
import numpy as np

from scipy.stats import norm as norm_gaussian

import numpy.typing as npt


from xaikd import bases, utils


class ExpectedU:
    @classmethod
    def compute(
        cls,
        arr_act: npt.NDArray,
        arr_ctx: npt.NDArray,
        arr_logodd: npt.NDArray,
        logodd_threshold: float,
    ):
        raise NotImplementedError()


def _test_analytic_basis(basis_name: str, compute_expected_U: ExpectedU):

    rng = np.random.default_rng(seed=1)
    n, d, num_locations = 100, 4, 20

    arr_act = rng.normal(size=(n, d, num_locations)) + 2
    arr_ctx = rng.normal(size=(n, d, num_locations)) + 2
    arr_logodd = rng.normal(size=(n,))
    logodd_threshold = 0.0

    basis = bases.get_basis(basis_name)

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        arr_logodd=arr_logodd,
        logodd_threshold=logodd_threshold,
    )

    expected_U = compute_expected_U.compute(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        arr_logodd=arr_logodd,
        logodd_threshold=logodd_threshold,
    )

    np.testing.assert_allclose(basis.U, expected_U)


def test_analytic_pca():
    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):
            arr_act = utils.flatten_3d_tensor(arr_act)
            cov = arr_act.T @ arr_act
            _, eigvecs = np.linalg.eigh(cov)
            return np.flip(eigvecs, axis=1)

    _test_analytic_basis("pca", Expected())


def test_analytic_gpca():
    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):
            arr_ctx = utils.flatten_3d_tensor(arr_ctx)
            cov = arr_ctx.T @ arr_ctx
            _, eigvecs = np.linalg.eigh(cov)
            return np.flip(eigvecs, axis=1)

    _test_analytic_basis("gradpca", Expected())


def test_analytic_prcasortabs():
    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):
            arr_act = utils.flatten_3d_tensor(arr_act)
            arr_ctx = utils.flatten_3d_tensor(arr_ctx)

            cov_ac = arr_act.T @ arr_ctx
            cov_acca = cov_ac + cov_ac.T

            eigvals, eigvecs = np.linalg.eigh(cov_acca)

            expected_U = eigvecs[:, np.argsort(-np.abs(eigvals))]
            return expected_U

    _test_analytic_basis("prcasortabs", Expected())


def test_analytic_prca():
    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):
            arr_act = utils.flatten_3d_tensor(arr_act)
            arr_ctx = utils.flatten_3d_tensor(arr_ctx)

            cov_ac = arr_act.T @ arr_ctx
            cov_acca = cov_ac + cov_ac.T

            _, eigvecs = np.linalg.eigh(cov_acca)

            eigvecs = np.flip(eigvecs, axis=1)

            return eigvecs

    _test_analytic_basis("prca", Expected())


def test_analytic_prcaposdef():
    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):
            arr_act = utils.flatten_3d_tensor(arr_act)
            arr_ctx = utils.flatten_3d_tensor(arr_ctx)

            cov_a = arr_act.T @ arr_act
            cov_c = arr_ctx.T @ arr_ctx

            tr_cov_a = np.trace(cov_a)
            tr_cov_c = np.trace(cov_c)

            cov_ac = arr_act.T @ arr_ctx
            cov_acca = cov_ac + cov_ac.T

            cov = (
                (1 / np.sqrt(tr_cov_a * tr_cov_c)) * cov_acca
                + (2 / tr_cov_a) * cov_a
                + (2 / tr_cov_c) * cov_c
            )

            _, eigvecs = np.linalg.eigh(cov)

            eigvecs = np.flip(eigvecs, axis=1)

            return eigvecs

    _test_analytic_basis("prcaposdef", Expected())


@pytest.mark.parametrize("percentile", [0.1, 1, 10, 50, 95])
def test_analytic_prcaposdef_weighting(percentile):

    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):

            y_pred = arr_logodd > logodd_threshold

            perc_pos = np.percentile(arr_logodd[y_pred == 1], percentile)
            perc_neg = np.percentile(arr_logodd[y_pred == 0], 100 - percentile)

            assert perc_pos > perc_neg

            std = 0.5 * (perc_pos - perc_neg)

            weights = norm_gaussian.pdf(
                arr_logodd,
                loc=logodd_threshold,
                scale=std,
            )

            weights = weights / np.sum(weights)

            arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))

            arr_act = utils.flatten_3d_tensor(arr_act)
            arr_ctx = utils.flatten_3d_tensor(arr_ctx)

            cov_a = arr_act.T @ arr_act
            cov_c = arr_ctx.T @ arr_ctx

            tr_cov_a = np.trace(cov_a)
            tr_cov_c = np.trace(cov_c)

            cov_ac = arr_act.T @ arr_ctx
            cov_acca = cov_ac + cov_ac.T

            cov = (
                (1 / np.sqrt(tr_cov_a * tr_cov_c)) * cov_acca
                + (2 / tr_cov_a) * cov_a
                + (2 / tr_cov_c) * cov_c
            )

            _, eigvecs = np.linalg.eigh(cov)

            eigvecs = np.flip(eigvecs, axis=1)

            return eigvecs

    _test_analytic_basis(f"prcaposdef-with-weight-p{percentile}", Expected())


@pytest.mark.parametrize("percentile", [1])
def test_analytic_gradpca_weighting(percentile):
    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):

            y_pred = arr_logodd > logodd_threshold

            perc_pos = np.percentile(arr_logodd[y_pred == 1], percentile)
            perc_neg = np.percentile(arr_logodd[y_pred == 0], 100 - percentile)

            assert perc_pos > perc_neg

            std = 0.5 * (perc_pos - perc_neg)

            weights = norm_gaussian.pdf(
                arr_logodd,
                loc=logodd_threshold,
                scale=std,
            )

            weights = weights / np.sum(weights)

            arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))

            arr_ctx = utils.flatten_3d_tensor(arr_ctx)

            cov_c = arr_ctx.T @ arr_ctx

            _, eigvecs = np.linalg.eigh(cov_c)

            eigvecs = np.flip(eigvecs, axis=1)

            return eigvecs

    _test_analytic_basis(f"gradpca-with-weight-p{percentile}", Expected())


@pytest.mark.parametrize(
    "basis_name",
    [
        "pca",
        "gradpca",
        "prcasortabs",
        "prcaposdef",
        "prcaposdef-with-weight-p1",
        "gradpca-with-weight-p1",
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
    actual = basis.scale_factors

    expected = np.mean((utils.flatten_3d_tensor(arr_act) @ U) ** 2, axis=0)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "basis_name,mat_func,criteria",
    [
        ("pca", lambda d: d[0].T @ d[0], lambda x: x),
        # ("pcacentering", lambda d: d[0].T @ d[0], lambda x: x),
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

    assert basis.centering == ("centering" in basis_name)

    if basis.centering:
        np.testing.assert_allclose(basis.mean, mean)
    else:
        np.testing.assert_allclose(basis.mean, np.zeros(d))
