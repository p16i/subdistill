import numpy as np


from scipy import stats

from .interface import OrthogonalBasis
from .register import register_basis

from xaikd import utils


@register_basis()
class PCA(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        N, _ = arr_act.shape

        cov = (arr_act.T @ arr_act) / N

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class PCARev(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        N, _ = arr_act.shape

        cov = (arr_act.T @ arr_act) / N

        _, eigvecs = utils.solve_eigh(cov)

        eigvecs = np.flip(eigvecs, axis=1)

        return eigvecs


@register_basis()
class Identity(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        _, d = arr_act.shape

        eigvecs = np.eye(d)

        return eigvecs
class Random(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        seed = kwargs["seed"]

        N, d = arr_act.shape
        rng = np.random.default_rng(seed=seed)

        U = stats.ortho_group(d).rvs(1, random_state=rng)

        return U


@register_basis()
class GradPCA(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        N, _ = arr_ctx.shape

        cov = (arr_ctx.T @ arr_ctx) / N

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class PRCASortAbs(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        N, _ = arr_act.shape

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        mean_ctx = np.mean(arr_ctx, axis=0)

        cov_ac = (arr_act.T @ arr_ctx) / N
        cov_acca = cov_ac + cov_ac.T

        _, eigvecs = utils.solve_eigh(cov_acca, sort_with_abs_eigvals=True)
        return eigvecs


@register_basis()
class PRCA(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        N, _ = arr_act.shape

        cov_ac = (arr_act.T @ arr_ctx) / N
        cov_acca = cov_ac + cov_ac.T

        _, eigvecs = utils.solve_eigh(cov_acca, sort_with_abs_eigvals=False)
        return eigvecs


@register_basis()
class PRCAPosDef(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        N, _ = arr_act.shape

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_a = (arr_act.T @ arr_act) / N
        tr_a = np.trace(cov_a)

        cov_c = (arr_ctx.T @ arr_ctx) / N
        tr_c = np.trace(cov_c)

        cov_ac = (arr_act.T @ arr_ctx) / N
        cov_acca = cov_ac + cov_ac.T

        coef_acca = 1
        coef_a = 2 * np.sqrt(tr_c / tr_a)
        coef_c = 2 * np.sqrt(tr_a / tr_c)

        print(
            f"Coefficients: coeff_acca={coef_acca:.4e}, coeff_a={coef_a:.4e}, coeff_c={coef_c:.4e} "
            + f"tr_a={tr_a:.4e}, tr_c={tr_c:.4e}"
        )

        cov_pos_def = coef_acca * cov_acca + coef_a * cov_a + coef_c * cov_c

        eigvals, eigvecs = utils.solve_eigh(cov_pos_def)

        min_eigval = np.min(eigvals)
        max_eigval = np.max(eigvals)
        print(f"range(eigvals)=[{min_eigval:.4e}, {max_eigval:.4e}]")

        # Mathematically, the eigenvalues should be non-negative.
        # Due to numerical stablity, there are situations that we have negative values.
        # In such cases, we tolerate and raise an error if the smallest value is relatetive large.
        if min_eigval < 0 and (np.abs(min_eigval) / max_eigval) > 1e-16:
            raise ValueError(
                f"We have {np.sum(eigvals < 0)} negative eigvals (the smallest one is {min_eigval:.4e})"
            )

        return eigvecs
