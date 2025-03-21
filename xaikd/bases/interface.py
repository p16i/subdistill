import typing
import numpy as np
from numpy import typing as npt

import torch
from torch import nn
from torch.nn import functional as F

from abc import ABC, abstractmethod
from xaikd import utils
from .adapter import AdapterMode, Adapter


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

        arr_act_on_U = arr_act @ U

        # todo: check that for PCA this is equal to eigenvalue
        output = np.mean((arr_act_on_U) ** 2, axis=0)

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
