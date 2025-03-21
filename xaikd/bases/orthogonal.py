import typing
import numpy.typing as npt

from abc import ABC, abstractmethod


import numpy as np
import torch


from .interface import OrthogonalBasis
from .register import register_basis
from .adapter import Adapter, AdapterMode

from xaikd import utils


@register_basis()
class PCA(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        arr_act = utils.flatten_3d_tensor(arr_act)

        cov = arr_act.T @ arr_act

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class GradPCA(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        cov = arr_ctx.T @ arr_ctx

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class PRCASortAbs(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        cov_ac = arr_act.T @ arr_ctx
        cov_acca = cov_ac + cov_ac.T

        _, eigvecs = utils.solve_eigh(cov_acca, sort_with_abs_eigvals=True)
        return eigvecs


@register_basis()
class PRCA(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_ac = arr_act.T @ arr_ctx
        cov_acca = cov_ac + cov_ac.T

        _, eigvecs = utils.solve_eigh(cov_acca, sort_with_abs_eigvals=False)
        return eigvecs


@register_basis()
class PRCAPosDef(OrthogonalBasis):
    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_a = arr_act.T @ arr_act
        tr_cov_a = np.trace(cov_a)

        cov_c = arr_ctx.T @ arr_ctx
        tr_cov_c = np.trace(cov_c)

        cov_ac = arr_act.T @ arr_ctx
        cov_acca = cov_ac + cov_ac.T

        cov_pos_def = (
            (1 / np.sqrt(tr_cov_a * tr_cov_c)) * cov_acca
            + (2 / tr_cov_a) * cov_a
            + (2 / tr_cov_c) * cov_c
        )

        eigvals, eigvecs = utils.solve_eigh(cov_pos_def)

        assert (eigvals >= 0).all()

        return eigvecs


@register_basis()
class PRCAPosDefSigmaASigmaC(OrthogonalBasis):
    @classmethod
    def slug(cls):
        return "prca-ablation-a-c"

    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_a = arr_act.T @ arr_act

        tr_cov_a = np.trace(cov_a)

        cov_c = arr_ctx.T @ arr_ctx
        tr_cov_c = np.trace(cov_c)

        cov = (2 / tr_cov_a) * cov_a + (2 / tr_cov_c) * cov_c
        eigvals, eigvecs = utils.solve_eigh(cov)

        assert (eigvals >= 0).all()

        return eigvecs


@register_basis()
class PRCAPosDefAblationSigmaASigmaAC(OrthogonalBasis):
    @classmethod
    def slug(cls):
        return "prca-ablation-a-ac"

    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        cov_a = arr_act.T @ arr_act
        tr_cov_a = np.trace(cov_a)

        tr_cov_c = np.trace(arr_ctx.T @ arr_ctx)

        cov_ac = arr_act.T @ arr_ctx
        cov_acca = cov_ac + cov_ac.T

        cov = (2 / tr_cov_a) * cov_a + (1 / np.sqrt(tr_cov_a * tr_cov_c)) * cov_acca
        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class PRCAPosDefAblationSigmaCSigmaAC(OrthogonalBasis):
    @classmethod
    def slug(cls):
        return "prca-ablation-c-ac"

    def _solve(self, arr_act, arr_ctx, arr_logodd, logodd_threshold, **kwargs):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)

        tr_cov_a = np.trace(arr_act.T @ arr_act)

        cov_c = arr_ctx.T @ arr_ctx
        tr_cov_c = np.trace(cov_c)

        cov_ac = arr_act.T @ arr_ctx
        cov_acca = cov_ac + cov_ac.T

        cov = (2 / tr_cov_c) * cov_c + (1 / np.sqrt(tr_cov_a * tr_cov_c)) * cov_acca
        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs
