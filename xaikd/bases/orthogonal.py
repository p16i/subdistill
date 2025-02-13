import typing
import numpy.typing as npt

from abc import ABC, abstractmethod


import numpy as np
import torch

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
    ) -> npt.NDArray:
        pass

    def construct_fh_rank_k_projection(self, k: int, device: str) -> typing.Callable:
        encoder = self.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device)
        decoder = self.construct_adapter(k=k, mode=AdapterMode.DECODER, device=device)

        def fh(mod, input, output):
            assert isinstance(output, torch.Tensor)
            return decoder(encoder(output))

        return fh

    def estimate_scale_factors(self, x: npt.NDArray, U: npt.NDArray) -> npt.NDArray:
        # remark: if centering (i.e., `mean(activation)=0`), then
        # this expresssion is `standard deviation`
        return np.mean((x @ U) ** 2, axis=0)

    def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
        U = torch.from_numpy(self.U[:, :k]).float()
        mean = torch.from_numpy(self.mean).float()

        return Adapter(U=U, mean=mean, mode=mode, device=device)

    def fit(self, arr_act, arr_ctx, **kwargs):
        _, d = arr_act.shape

        if self.centering:
            mean = np.mean(arr_act, axis=0)
            arr_centered_arr = arr_act - mean
        else:
            mean = np.zeros(d)
            # remark: the name might be a bit confusing
            arr_centered_arr = arr_act

        self._U = self._solve(arr_act=arr_centered_arr, arr_ctx=arr_ctx)

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
class Identity(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        _, d = arr_act.shape

        return np.eye(d)


@register_basis()
class Random(OrthogonalBasis):

    def fit(self, arr_act, arr_ctx, **kwargs):
        assert "seed" in kwargs, "please specify `seed`"

        seed = kwargs["seed"]

        self.rng = np.random.default_rng(seed=seed)
        super().fit(arr_act, arr_ctx, **kwargs)

    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        _, d = arr_act.shape
        U = ortho_group.rvs(dim=d, random_state=self.rng)

        return U


@register_basis()
class PCA(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_act

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class GradPCA(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_ctx.T @ arr_ctx

        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class PRCASortAbs(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        _, eigvecs = utils.solve_eigh(cov, sort_with_abs_eigvals=True)
        return eigvecs


@register_basis()
class PRCA(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        _, eigvecs = utils.solve_eigh(cov, sort_with_abs_eigvals=False)
        return eigvecs


@register_basis()
class PRCAPosDef(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):

        cov_a = arr_act.T @ arr_act

        tr_cov_a = np.trace(cov_a)

        cov_c = arr_ctx.T @ arr_ctx
        tr_cov_c = np.trace(cov_c)

        cov_ac = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        cov_pos_def = (
            (2 / tr_cov_a) * cov_a
            + (2 / tr_cov_c) * cov_c
            + (1 / np.power(tr_cov_a * tr_cov_c, 0.5)) * cov_ac
        )
        eigvals, eigvecs = utils.solve_eigh(cov_pos_def)

        assert (eigvals >= 0).all()

        return eigvecs


@register_basis()
class PRCAPosDefSigmaASigmaC(OrthogonalBasis):
    @classmethod
    def slug(cls):
        return "prca-ablation-a-c"

    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):

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

    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):

        cov_a = arr_act.T @ arr_act

        tr_cov_a = np.trace(cov_a)

        cov_c = arr_ctx.T @ arr_ctx
        tr_cov_c = np.trace(cov_c)

        cov_ac = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        cov = (2 / tr_cov_a) * cov_a + (1 / np.power(tr_cov_a * tr_cov_c, 0.5)) * cov_ac
        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


@register_basis()
class PRCAPosDefAblationSigmaCSigmaAC(OrthogonalBasis):
    @classmethod
    def slug(cls):
        return "prca-ablation-c-ac"

    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):

        cov_a = arr_act.T @ arr_act

        tr_cov_a = np.trace(cov_a)

        cov_c = arr_ctx.T @ arr_ctx
        tr_cov_c = np.trace(cov_c)

        cov_ac = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        cov = (2 / tr_cov_c) * cov_c + (1 / np.power(tr_cov_a * tr_cov_c, 0.5)) * cov_ac
        _, eigvecs = utils.solve_eigh(cov)

        return eigvecs


# @register_basis("pcalookahead")
# class PCALookAhead(Orthogonal):
#     def fit(self, arr_act, arr_ctx, **kwargs):
#         assert self.centering == False, "we only support `uncentered` version` for now"

#         self.model = kwargs["model"]
#         self.layer = kwargs["layer"]
#         self.dataloader = kwargs["dataloader"]
#         self.arr_act = arr_act

#         self._cache = dict()
#         self.U = self._get_initialization(
#             model=self.model, arr_act=arr_act, arr_ctx=arr_ctx
#         )

#     def _get_initialization(
#         self, model: nn.Module, arr_act: npt.NDArray, arr_ctx: npt.NDArray
#     ) -> npt.NDArray:
#         if isinstance(model, models.vit.VisionTransformer):
#             ref_basis = PCA()
#         else:
#             ref_basis = PRCASortAbs()
#         ref_basis.fit(arr_act=arr_act, arr_ctx=arr_ctx)

#         return ref_basis.U

#     def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
#         assert self.centering == False, "we only support `uncetered` version"

#         if not k in self._cache:

#             Uinit = self.U[:, :k].copy()

#             U = pcalookahead.fit(
#                 model=self.model,
#                 layer=self.layer,
#                 dataloader=self.dataloader,
#                 Uinit=Uinit,
#                 k=k,
#                 verbose=False,
#                 device=device,
#             )
#             scale = self._estimate_scale_factor(self.arr_act, U)
#             self._cache[k] = (U, scale)
#         else:
#             U, scale = self._cache[k]

#         d, k = U.shape

#         return Adapter(
#             U=torch.from_numpy(U).float(),
#             mean=torch.zeros(d).float(),
#             mode=mode,
#             device=device,
#         )

#     def get_scale_factors_for_k(self, k: int) -> npt.NDArray:
#         _, scale = self._cache[k]

#         return scale
