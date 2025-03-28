import typing
import numpy.typing as npt

from abc import ABC, abstractmethod


import numpy as np
import torch


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
        print(f"range(eigvals)=[{np.min(eigvals):.4e}, {np.max(eigvals):.4e}]")

        assert (eigvals >= 0).all()

        return eigvecs
