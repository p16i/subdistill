from numpy._typing import NDArray
import pytest
import typing
import numpy as np

from scipy.stats import norm as norm_gaussian

import numpy.typing as npt


from xaikd import bases, utils


def _generate_dummy_act_ctx(
    num_datapoints: int, num_channels: int, num_spatials: int, seed: int
) -> typing.Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

    rng = np.random.default_rng(seed=seed)
    activation = rng.random(
        size=(num_datapoints, num_channels, num_spatials)
    ) + rng.random(size=(1,))
    context = rng.random(size=(num_datapoints, num_channels, num_spatials))

    mean = np.mean(utils.flatten_3d_tensor(activation), axis=0)
    activation -= mean[None, :, None]

    return activation, context, mean


class ExpectedU:
    @classmethod
    def compute(
        cls,
        arr_act: npt.NDArray,
        arr_ctx: npt.NDArray,
        mean_act: npt.NDArray,
        arr_logodd: npt.NDArray,
        logodd_threshold: float,
    ):
        raise NotImplementedError()


def _test_analytic_basis(basis_name: str, compute_expected_U: ExpectedU):

    rng = np.random.default_rng(seed=1)
    n, d, num_locations = 100, 4, 20

    arr_act = rng.normal(size=(n, d, num_locations)) + 2
    arr_ctx = rng.normal(size=(n, d, num_locations)) + 2

    mean_act = np.mean(utils.flatten_3d_tensor(arr_act), axis=0)
    arr_act -= mean_act[None, :, None]

    arr_logodd = rng.normal(size=(n,))
    logodd_threshold = 0.0

    basis = bases.get_basis(basis_name)

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        mean_act=mean_act,
        arr_logodd=arr_logodd,
        logodd_threshold=logodd_threshold,
        strict_mode=True,
    )

    expected_U = compute_expected_U.compute(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        mean_act=mean_act,
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
            mean_act: npt.NDArray,
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
            mean_act: npt.NDArray,
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
            mean_act: npt.NDArray,
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
            mean_act: npt.NDArray,
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
            mean_act: npt.NDArray,
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


@pytest.mark.parametrize("entropy_ratio", [0.5, 0.95, 1.0])
def test_analytic_prcaposdef_weighting(entropy_ratio):

    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            mean_act: npt.NDArray,
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):

            std = bases.orthogonal_weighting.estimate_std_wrt_ratio_maxent(
                arr_logits=arr_logodd, threshold=logodd_threshold, ratio=entropy_ratio
            )

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

    _test_analytic_basis(f"prcaposdef-entropy{entropy_ratio}", Expected())


@pytest.mark.parametrize("entropy_ratio", [0.95])
def test_analytic_gradpca_weighting(entropy_ratio):
    class Expected(ExpectedU):
        @classmethod
        def compute(
            cls,
            arr_act: np.ndarray[typing.Any, np.dtype],
            arr_ctx: np.ndarray[typing.Any, np.dtype],
            mean_act: npt.NDArray,
            arr_logodd: np.ndarray[typing.Any, np.dtype],
            logodd_threshold: float,
        ):

            std = bases.orthogonal_weighting.estimate_std_wrt_ratio_maxent(
                arr_logits=arr_logodd, threshold=logodd_threshold, ratio=entropy_ratio
            )

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

    _test_analytic_basis(f"gradpca-entropy{entropy_ratio}", Expected())


@pytest.mark.parametrize(
    "basis_name",
    [
        "pca",
        "gradpca",
        "prcaposdef",
        "prcaposdef-entropy0.95",
        "gradpca-entropy0.95",
    ],
)
def test_correct_scale_orthogoal_bases(basis_name):
    np.random.seed(1)
    n, d, num_locations = 10, 5, 20
    arr_act, arr_ctx, mean_act = _generate_dummy_act_ctx(
        num_datapoints=n, num_channels=d, num_spatials=num_locations, seed=1
    )

    arr_logodd = np.random.randn(n)
    logodd_threshold = 0.0

    basis = bases.get_basis(f"{basis_name}")

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        mean_act=mean_act,
        arr_logodd=arr_logodd,
        logodd_threshold=logodd_threshold,
        seed=1,
        strict_mode=True,
    )

    U = basis.U
    actual = basis.scale_factors

    arr_flattened_act = utils.flatten_3d_tensor(arr_act)
    if basis.centering:
        arr_flattened_act -= np.mean(arr_flattened_act, axis=0)

    expected = np.mean((arr_flattened_act @ U) ** 2, axis=0)
    np.testing.assert_allclose(actual, expected)

    if basis_name == "pca":
        eigvals, _ = utils.solve_eigh(
            arr_flattened_act.T @ arr_flattened_act / (n * num_locations)
        )
        np.testing.assert_allclose(actual, eigvals)


def _solve_pca(arr_logodd, arr_act, arr_ctx):
    arr_act = utils.flatten_3d_tensor(arr_act)

    cov = arr_act.T @ arr_act
    return utils.solve_eigh(cov)


def _solve_gradpca(arr_logodd, arr_act, arr_ctx):
    arr_ctx = utils.flatten_3d_tensor(arr_ctx)

    cov = arr_ctx.T @ arr_ctx
    return utils.solve_eigh(cov)


def _solve_gradpca_threshold_0_entropy_0_95(arr_logodd, arr_act, arr_ctx):
    std = bases.orthogonal_weighting.estimate_std_wrt_ratio_maxent(
        arr_logits=arr_logodd, threshold=0, ratio=0.95
    )
    weights = norm_gaussian.pdf(
        arr_logodd,
        loc=0,
        scale=std,
    )
    weights = weights / np.sum(weights)
    arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))
    arr_ctx = utils.flatten_3d_tensor(arr_ctx)

    cov = arr_ctx.T @ arr_ctx
    return utils.solve_eigh(cov)


def _solve_prca_posdef(arr_logodd, arr_act, arr_ctx):
    arr_act = utils.flatten_3d_tensor(arr_act)
    arr_ctx = utils.flatten_3d_tensor(arr_ctx)

    cov_a = arr_act.T @ arr_act
    cov_c = arr_ctx.T @ arr_ctx
    cov_ac = arr_act.T @ arr_ctx
    cov_acca = cov_ac + cov_ac.T

    tr_a = np.trace(cov_a)
    tr_c = np.trace(cov_c)

    cov = (2 / tr_a) * cov_a + (2 / tr_c) * cov_c + cov_acca / np.sqrt(tr_a * tr_c)

    return utils.solve_eigh(cov)


def _solve_prca_posdef_as_defined_in_paper(arr_logodd, arr_act, arr_ctx):
    arr_act = utils.flatten_3d_tensor(arr_act)
    arr_ctx = utils.flatten_3d_tensor(arr_ctx)

    cov_a = arr_act.T @ arr_act
    cov_c = arr_ctx.T @ arr_ctx
    cov_ac = arr_act.T @ arr_ctx
    cov_acca = cov_ac + cov_ac.T

    tr_a = np.trace(cov_a)
    tr_c = np.trace(cov_c)

    cov = cov_acca + 2 * np.sqrt(tr_c / tr_a) * cov_a + 2 * np.sqrt(tr_a / tr_c) * cov_c

    return utils.solve_eigh(cov)


def _solve_prca_posdef_threshold_0_entropy_0_95(arr_logodd, arr_act, arr_ctx):

    std = bases.orthogonal_weighting.estimate_std_wrt_ratio_maxent(
        arr_logits=arr_logodd, threshold=0, ratio=0.95
    )
    weights = norm_gaussian.pdf(
        arr_logodd,
        loc=0,
        scale=std,
    )
    weights = weights / np.sum(weights)

    arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))

    arr_act = utils.flatten_3d_tensor(arr_act)
    arr_ctx = utils.flatten_3d_tensor(arr_ctx)

    cov_a = arr_act.T @ arr_act
    cov_c = arr_ctx.T @ arr_ctx
    cov_ac = arr_act.T @ arr_ctx
    cov_acca = cov_ac + cov_ac.T

    tr_a = np.trace(cov_a)
    tr_c = np.trace(cov_c)

    cov = (2 / tr_a) * cov_a + (2 / tr_c) * cov_c + cov_acca / np.sqrt(tr_a * tr_c)

    return utils.solve_eigh(cov)


@pytest.mark.parametrize(
    "basis_name,solve_func",
    [
        ("pca", _solve_pca),
        ("gradpca", _solve_gradpca),
        ("prcaposdef", _solve_prca_posdef),
        ("prcaposdef", _solve_prca_posdef_as_defined_in_paper),
        ("prcaposdef-entropy0.95", _solve_prca_posdef_threshold_0_entropy_0_95),
        ("gradpca-entropy0.95", _solve_gradpca_threshold_0_entropy_0_95),
    ],
)
def test_centering_orthogonal_bases(basis_name, solve_func):
    np.random.seed(1)
    n, d, num_locations = 10, 5, 20
    basis = bases.get_basis(basis_name)

    activation, context, mean_act = _generate_dummy_act_ctx(
        num_datapoints=n, num_channels=d, num_spatials=num_locations, seed=1
    )

    arr_logodd = np.random.randn(
        n,
    )
    threshold = 0

    assert basis.centering

    expected_eigvals, expected_U = solve_func(
        arr_logodd,
        activation,
        context,
    )
    basis.fit(
        arr_act=activation,
        arr_ctx=context,
        mean_act=mean_act,
        arr_logodd=arr_logodd,
        logodd_threshold=threshold,
        device="cpu",
        strict_mode=True,
    )

    np.testing.assert_allclose(basis.mean, mean_act)

    actual_U = basis.U

    np.testing.assert_allclose(actual_U, expected_U)

    if basis.centering:
        assert basis.mean.shape == (d,)
        np.testing.assert_allclose(basis.mean, mean_act)
    else:
        np.testing.assert_allclose(basis.mean, np.zeros(d))


def test_equivalence_between_prcaposdef_and_prcposdef_uniform():
    n, d, num_locations = 10, 5, 20

    activation = np.random.rand(n, d, num_locations)
    context = np.random.rand(n, d, num_locations)
    activation, context, mean_act = _generate_dummy_act_ctx(
        num_datapoints=n, num_channels=d, num_spatials=num_locations, seed=1
    )
    arr_logodd = np.random.randn(
        n,
    )
    threshold = 0

    prcaposdef = bases.PRCAPosDef()
    prcaposdef.fit(
        arr_act=activation,
        arr_ctx=context,
        mean_act=mean_act,
        arr_logodd=arr_logodd,
        logodd_threshold=threshold,
    )

    prcaposdef_uniform = bases.PRCAPosDefUniformWeight()
    prcaposdef_uniform.fit(
        arr_act=activation,
        arr_ctx=context,
        mean_act=mean_act,
        arr_logodd=arr_logodd,
        logodd_threshold=threshold,
    )

    np.testing.assert_allclose(
        prcaposdef.U,
        prcaposdef_uniform.U,
    )

    np.testing.assert_allclose(
        prcaposdef.scale_factors, prcaposdef_uniform.scale_factors
    )
