import typing
import numpy.typing as npt

from abc import ABC, abstractmethod


import numpy as np
import torch

from scipy.stats import norm as norm_gaussian
from scipy.stats import ortho_group

from .register import register_basis
from .adapter import Adapter, AdapterMode

from xaikd import utils


class OrthogonalBasis(ABC):
    def __init__(self, centering: bool = False):
        self.centering = centering

    @property
    def mean(self) -> npt.NDArray:
        return self._mean

    @property
    def U(self) -> npt.NDArray:
        return self._U

    @property
    def scale_factors(self) -> npt.NDArray:
        return self._scale_factors

    @abstractmethod
    def _solve(
        self,
        arr_act: npt.NDArray,
        arr_ctx: npt.NDArray,
        arr_logodd: npt.NDArray,
        logodd_threshold: float,
    ) -> npt.NDArray:
        pass

    def construct_fh_rank_k_projection(self, k: int, device: str) -> typing.Callable:
        encoder = self.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device)
        decoder = self.construct_adapter(k=k, mode=AdapterMode.DECODER, device=device)

        def fh(mod, input, output):
            assert isinstance(output, torch.Tensor)
            return decoder(encoder(output))

        return fh

    def estimate_scale_factors(
        self, arr_act: npt.NDArray, U: npt.NDArray
    ) -> npt.NDArray:
        # remark: if centering (i.e., `mean(activation)=0`), then
        # this expresssion is `standard deviation`

        u1 = U[:, 0]

        # we do this to make sure that it compatible with the basis
        arr_act = utils.flatten_3d_tensor(arr_act)
        output = np.array([np.mean((arr_act @ u1) ** 2)])

        return output

    def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
        U = torch.from_numpy(self.U[:, :k]).float()
        mean = torch.from_numpy(self.mean).float()

        return Adapter(U=U, mean=mean, mode=mode, device=device)

    def fit(
        self,
        arr_act: npt.NDArray,
        arr_ctx: npt.NDArray,
        arr_logodd: npt.NDArray,
        logodd_threshold: float,
        **kwargs,
    ):
        _, d, _ = arr_act.shape

        if self.centering:
            mean = np.mean(arr_act, axis=0)
            arr_centered_arr = arr_act - mean
        else:
            mean = np.zeros(d)
            # remark: the name might be a bit confusing
            arr_centered_arr = arr_act

        self._U = self._solve(
            arr_act=arr_centered_arr,
            arr_ctx=arr_ctx,
            arr_logodd=arr_logodd,
            logodd_threshold=logodd_threshold,
        )

        self._scale_factors = self.estimate_scale_factors(arr_centered_arr, self._U)

        self._mean = mean

    def get_Uk(self, k: int) -> npt.NDArray[np.float32]:
        return self.U[:, :k]

    def get_scale_factors_for_k(self, k: int) -> npt.NDArray[np.float32]:
        return self.scale_factors[:k]

    @classmethod
    def slug(cls):
        return cls.__name__.lower()


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
