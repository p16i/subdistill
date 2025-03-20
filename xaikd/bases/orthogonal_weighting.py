import numpy as np
from numpy import typing as npt

from scipy.stats import norm as norm_gaussian

from xaikd import utils

from .orthogonal import OrthogonalBasis

from .register import register_basis


class PRCAPosDefWeightSTDWithPercentile(OrthogonalBasis):
    percentile = None

    def _compute_sample_weight(
        self, arr_logodd: npt.NDArray, logodd_threshold: float
    ) -> npt.NDArray:
        y_pred = arr_logodd > logodd_threshold
        assert self.percentile is not None

        perc_pos = np.percentile(arr_logodd[y_pred == 1], self.percentile)
        perc_neg = np.percentile(arr_logodd[y_pred == 0], 100 - self.percentile)

        assert perc_pos > perc_neg

        std = 0.5 * (perc_pos - perc_neg)

        weights = norm_gaussian.pdf(
            arr_logodd,
            loc=logodd_threshold,
            scale=std,
        )

        weights = weights / np.sum(weights)

        assert weights.shape == arr_logodd.shape

        return weights

    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        weights = self._compute_sample_weight(
            arr_logodd=arr_logodd, logodd_threshold=logodd_threshold
        )
        arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))

        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_a = arr_act.T @ arr_act
        tr_cov_a = np.trace(cov_a)

        cov_c = arr_ctx.T @ arr_ctx
        tr_cov_c = np.trace(cov_c)

        cov_ac = arr_act.T @ arr_ctx
        cov_acca = cov_ac + cov_ac.T

        cov = (
            (1 / np.sqrt(tr_cov_a * tr_cov_c)) * cov_acca
            + (2 / tr_cov_a) * cov_a
            + (2 / tr_cov_c) * cov_c
        )

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs

    @classmethod
    def slug(cls):
        return f"prcaposdef-with-weight-p{cls.percentile}"


@register_basis()
class PRCAPosDefWeightSTDWithP0_1(PRCAPosDefWeightSTDWithPercentile):
    percentile = 0.1


@register_basis()
class PRCAPosDefWeightSTDWithP1(PRCAPosDefWeightSTDWithPercentile):
    percentile = 1


@register_basis()
class PRCAPosDefWeightSTDWithP10(PRCAPosDefWeightSTDWithPercentile):
    percentile = 10


@register_basis()
class PRCAPosDefWeightSTDWithP50(PRCAPosDefWeightSTDWithPercentile):
    percentile = 50


@register_basis()
class PRCAPosDefWeightSTDWithP95(PRCAPosDefWeightSTDWithPercentile):
    percentile = 95


@register_basis()
class GradPCAWeightSTDWithPercentile(OrthogonalBasis):
    percentile = 1

    def _compute_sample_weight(
        self, arr_logodd: npt.NDArray, logodd_threshold: float
    ) -> npt.NDArray:
        y_pred = arr_logodd > logodd_threshold
        assert self.percentile is not None

        perc_pos = np.percentile(arr_logodd[y_pred == 1], self.percentile)
        perc_neg = np.percentile(arr_logodd[y_pred == 0], 100 - self.percentile)

        assert perc_pos > perc_neg

        std = 0.5 * (perc_pos - perc_neg)

        weights = norm_gaussian.pdf(
            arr_logodd,
            loc=logodd_threshold,
            scale=std,
        )

        weights = weights / np.sum(weights)

        assert weights.shape == arr_logodd.shape

        return weights

    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        weights = self._compute_sample_weight(
            arr_logodd=arr_logodd, logodd_threshold=logodd_threshold
        )
        arr_ctx = arr_ctx * weights.reshape((-1, 1, 1))

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_c = arr_ctx.T @ arr_ctx

        _, eigvecs = utils.solve_eigh(cov_c)

        return eigvecs

    @classmethod
    def slug(cls):
        return f"gradpca-with-weight-p{cls.percentile}"
