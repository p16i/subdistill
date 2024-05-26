import os
import typing
import numpy as np
from numpy._typing import NDArray
import numpy.typing as npt

from pathlib import Path

from scipy.stats import ortho_group


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from abc import ABC

from . import pcalookahead


from enum import Enum

from xaikd.bases import pcalookahead

EPS = 1e-6
BASES = dict()

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


def register_basis(name):
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        """Wrapped function to register a data modality provider with name `name`"""
        BASES[name] = cls

        return cls

    return wrapped


class Basis(ABC):
    U: npt.NDArray
    scale: npt.NDArray
    mean: npt.NDArray

    def __init__(self, centering: bool = False):
        self.centering = centering

    def fit(self, arr_act: npt.NDArray, arr_ctx: npt.NDArray, **kwargs):
        raise NotImplementedError("...")

    def _estimate_scale_factor(self, x: npt.NDArray, U: npt.NDArray) -> npt.NDArray:
        # remark: if centering (i.e., `mean(activation)=0`), then
        # this expresssion is `standard deviation`
        return np.mean((x @ U) ** 2, axis=0)

    def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
        U = torch.from_numpy(self.U[:, :k])
        mean = torch.from_numpy(self.mean)

        return Adapter(U=U, mean=mean, mode=mode, device=device)

    def construct_fh_rank_k_projection(self, k: int, device: str) -> typing.Callable:
        encoder = self.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device)
        decoder = self.construct_adapter(k=k, mode=AdapterMode.DECODER, device=device)

        def fh(mod, input, output):
            assert isinstance(output, torch.Tensor)
            return decoder(encoder(output))

        return fh

    def __str__(self) -> str:
        return getattr(self, "__name")

    def get_scale_factors_for_k(self, k: int) -> npt.NDArray:
        return self.scale[:k]


def get_basis(slug, **kwargs) -> Basis:
    name_slug, centering_slug = slug.split("--")
    centering = True if centering_slug == "centered" else False

    if name_slug == ["random"]:
        assert "seed" in kwargs, "`seed` must be specify for `random` basis."

    assert centering_slug in ["uncentered", "centered"], f"Value `{centering_slug}`"

    basis = BASES[name_slug](centering=centering, **kwargs)

    setattr(basis, "__name", slug)

    return basis


class Orthogonal(Basis):
    def fit(self, arr_act, arr_ctx, **kwargs):
        _, d = arr_act.shape

        if self.centering:
            mean = np.mean(arr_act, axis=0)
            arr_centered_arr = arr_act - mean
        else:
            mean = np.zeros(d)
            # remark: the name might be a bit confusing
            arr_centered_arr = arr_act

        U = self._solve(arr_act=arr_centered_arr, arr_ctx=arr_ctx)

        scale = self._estimate_scale_factor(arr_centered_arr, U)

        self.U = U
        self.scale = scale
        self.mean = mean

    def _solve(
        self,
        arr_act: npt.NDArray,
        arr_ctx: npt.NDArray,
    ) -> npt.NDArray:
        # remark: if centerining = true, then arr_act is already centered.
        raise NotImplementedError("...")


@register_basis("identity")
class Identity(Orthogonal):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        _, d = arr_act.shape

        return np.eye(d)


@register_basis("pca")
class PCA(Orthogonal):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_act

        eigvals, eigvecs = np.linalg.eigh(cov)

        sorted_ix = np.argsort(-eigvals)

        U = eigvecs[:, sorted_ix].copy()

        return U


@register_basis("prca-sortabs")
class PRCASortAbs(Orthogonal):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        eigvals, eigvecs = np.linalg.eigh(cov)

        sorted_ix = np.argsort(-np.abs(eigvals))

        U = eigvecs[:, sorted_ix].copy()

        return U


@register_basis("prca")
class PRCA(Orthogonal):
    def _solve(
        self,
        arr_act,
        arr_ctx,
    ):
        cov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

        eigvals, eigvecs = np.linalg.eigh(cov)

        sorted_ix = np.argsort(-eigvals)

        U = eigvecs[:, sorted_ix].copy()

        return U


@register_basis("pcalookahead")
class PCALookAhead(PCA):
    def fit(self, arr_act, arr_ctx, **kwargs):
        assert self.centering == False, "we only support `uncentered` version` for now"

        super().fit(arr_act=arr_act, arr_ctx=arr_ctx)

        self.model = kwargs["model"]
        self.layer = kwargs["layer"]
        self.dataloader = kwargs["dataloader"]
        self.arr_act = arr_act

        self._cache = dict()

    def construct_adapter(self, k: int, mode: AdapterMode, device: str) -> Adapter:
        assert self.centering == False, "we only support `uncetered` version"

        if not k in self._cache:

            # this is the U from PCA
            Uinit = self.U[:, :k].copy()

            U = pcalookahead.fit(
                model=self.model,
                layer=self.layer,
                dataloader=self.dataloader,
                Uinit=Uinit,
                k=k,
                verbose=False,
                device=device,
            )
            scale = self._estimate_scale_factor(self.arr_act, U)
            self._cache[k] = (U, scale)
        else:
            U, scale = self._cache[k]

        d, k = U.shape

        return Adapter(
            U=torch.from_numpy(U),
            mean=torch.zeros(d),
            mode=mode,
            device=device,
        )

    def get_scale_factors_for_k(self, k: int) -> npt.NDArray:
        _, scale = self._cache[k]

        return scale
