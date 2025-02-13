import os
import typing
from functools import partial
import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt

from pathlib import Path

from scipy.stats import ortho_group


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from . import pcalookahead


from enum import Enum

from xaikd.bases import pcalookahead
from xaikd.bases.learners import (
    PRCAGreedyLearner,
    PRCAReconGreedy,
    PRCASignAlignGreedy,
    PRCASignAlignGreedyV2,
)
from xaikd import models


EPS = 1e-6
BASES = dict()


def _solve_eigh(
    cov: npt.NDArray, sort_with_abs_eigvals=False
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    eigvals, eigvecs = np.linalg.eigh(cov)

    assert len(eigvals.shape) == 1

    if sort_with_abs_eigvals:
        eigvals = np.abs(eigvals)

    # we sort in descending order
    indices = np.argsort(-eigvals)
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:, indices]

    return eigvals, eigvecs


AdapterMode = Enum("AdapterMode", ["ENCODER", "DECODER"])


class Adapter(torch.nn.Module):
    def __init__(
        self,
        U: torch.Tensor,
        mean: torch.Tensor,
        device: str,
        mode: AdapterMode,
    ) -> None:
        super().__init__()

        d, k = U.shape

        assert mean.shape[0] == d

        self.mat_encoder = U.T.unsqueeze(2).unsqueeze(3).to(device)
        self.mat_decoder = U.unsqueeze(2).unsqueeze(3).to(device)

        self.mean = mean.reshape((1, -1, 1, 1)).to(device)

        self.mode = mode

    def forward(self, x) -> torch.Tensor:
        if self.mode == AdapterMode.ENCODER:
            return self.encode(x)
        elif self.mode == AdapterMode.DECODER:
            return self.decode(x)
        else:
            raise ValueError(f"[mode={self.mode}] doesn't exist!")

    def encode(self, x):
        x = x - self.mean
        x = F.conv2d(x, self.mat_encoder)
        return x

    def decode(self, x):
        x = F.conv2d(x, self.mat_decoder)
        x = x + self.mean
        return x


def register_basis():
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        """Wrapped function to register a data modality provider with name `name`"""

        slug = cls.slug()

        assert not (slug in BASES)

        BASES[slug] = cls

        return cls

    return wrapped


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

    def get_Uk(self, k: int) -> NDArray[np.float32]:
        return self.U[:, :k]

    def get_scale_factors_for_k(self, k: int) -> NDArray[np.float32]:
        return self.scale_factors[:k]

    @classmethod
    def slug(cls):
        return cls.__name__.lower()


def get_basis(basis_name, **kwargs) -> OrthogonalBasis:

    basis = BASES[basis_name](**kwargs)

    return basis


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

        _, eigvecs = _solve_eigh(cov)

        return eigvecs


@register_basis()
class GradPCA(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_ctx.T @ arr_ctx

        _, eigvecs = _solve_eigh(cov)

        return eigvecs


@register_basis()
class PRCASortAbs(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        _, eigvecs = _solve_eigh(cov, sort_with_abs_eigvals=True)
        return eigvecs


@register_basis()
class PRCA(OrthogonalBasis):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        _, eigvecs = _solve_eigh(cov, sort_with_abs_eigvals=False)
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
        eigvals, eigvecs = _solve_eigh(cov_pos_def)

        assert (eigvals >= 0).all()

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


def _add_centering_variants():

    for base_variant_cls in [PCA]:
        base_variant_slug = base_variant_cls.slug()
        slug = f"{base_variant_slug}centering"

        assert not (slug in BASES)
        BASES[slug] = partial(base_variant_cls, centering=True)


_add_centering_variants()
