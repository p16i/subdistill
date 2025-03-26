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
    def _solve(
        self, arr_act, arr_ctx, mean_act, arr_logodd, logodd_threshold, **kwargs
    ):
        arr_act = utils.flatten_3d_tensor(arr_act)
        N, _ = arr_act.shape

        cov = (arr_act.T @ arr_act) / N - np.outer(mean_act, mean_act)

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class GradPCA(OrthogonalBasis):
    def _solve(
        self, arr_act, arr_ctx, mean_act, arr_logodd, logodd_threshold, **kwargs
    ):

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        N, _ = arr_ctx.shape

        cov = (arr_ctx.T @ arr_ctx) / N

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class PRCASortAbs(OrthogonalBasis):
    def _solve(
        self, arr_act, arr_ctx, mean_act, arr_logodd, logodd_threshold, **kwargs
    ):
        arr_act = utils.flatten_3d_tensor(arr_act)
        N, _ = arr_act.shape

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        mean_ctx = np.mean(arr_ctx, axis=0)

        cov_ac = (arr_act.T @ arr_ctx) / N - np.outer(mean_act, mean_ctx)
        cov_acca = cov_ac + cov_ac.T

        _, eigvecs = utils.solve_eigh(cov_acca, sort_with_abs_eigvals=True)
        return eigvecs


@register_basis()
class PRCA(OrthogonalBasis):
    def _solve(
        self, arr_act, arr_ctx, mean_act, arr_logodd, logodd_threshold, **kwargs
    ):
        arr_act = utils.flatten_3d_tensor(arr_act)
        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        mean_ctx = np.mean(arr_ctx, axis=0)
        N, _ = arr_act.shape

        cov_ac = (arr_act.T @ arr_ctx) / N - np.outer(mean_act, mean_ctx)
        cov_acca = cov_ac + cov_ac.T

        _, eigvecs = utils.solve_eigh(cov_acca, sort_with_abs_eigvals=False)
        return eigvecs


@register_basis()
class PRCAPosDef(OrthogonalBasis):
    def _solve(
        self, arr_act, arr_ctx, mean_act, arr_logodd, logodd_threshold, **kwargs
    ):
        arr_act = utils.flatten_3d_tensor(arr_act)
        N, _ = arr_act.shape

        arr_ctx = utils.flatten_3d_tensor(arr_ctx)
        mean_ctx = np.mean(arr_ctx, axis=0)

        cov_a = (arr_act.T @ arr_act) / N - np.outer(mean_act, mean_act)
        tr_a = np.trace(cov_a)

        cov_c = (arr_ctx.T @ arr_ctx) / N
        tr_c = np.trace(cov_c)

        cov_ac = (arr_act.T @ arr_ctx) / N - np.outer(mean_act, mean_ctx)
        cov_acca = cov_ac + cov_ac.T

        cov_pos_def = (
            cov_acca
            + (2 * np.sqrt(tr_c / tr_a)) * cov_a
            + (2 * np.sqrt(tr_a / tr_c)) * cov_c
        )
        eigvals, eigvecs = utils.solve_eigh(cov_pos_def)

        assert (eigvals >= 0).all()

        return eigvecs
