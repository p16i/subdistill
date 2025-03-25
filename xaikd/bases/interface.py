import typing
import numpy as np
from numpy import typing as npt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm

from abc import ABC, abstractmethod
from xaikd import utils
from .adapter import AdapterMode, Adapter

from torchmetrics import MeanMetric


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

    @torch.no_grad()
    def estimate_scale_factors(
        self, arr_act: npt.NDArray, U: npt.NDArray, device=utils.get_device()
    ) -> npt.NDArray:

        d, _ = U.shape

        # remark: if centering (i.e., `mean(activation)=0`), then
        # this expresssion is `standard deviation`

        # we do this to make sure that it compatible with the basis
        arr_act = utils.flatten_3d_tensor(arr_act)

        dl = DataLoader(
            TensorDataset(torch.from_numpy(arr_act).float()),
            batch_size=1024,
            shuffle=False,
            persistent_workers=True,
        )

        ts_U = torch.from_numpy(U).float().to(device)

        arr_scale_factors = []
        for i in tqdm(
            range(d),
            desc=f"[basis={self.__class__.slug()}] estimating scale factors",
        ):

            ui = ts_U[:, i]

            scale = MeanMetric()

            for bx in dl:
                bx = bx[0].to(device)
                bx_on_ui = bx @ ui
                b_scale = (bx_on_ui**2).cpu().numpy()
                scale.update(b_scale)

            scale = scale.compute()
            arr_scale_factors.append(scale)

        arr_scale_factors = np.array(arr_scale_factors)

        return arr_scale_factors

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
            mean = np.mean(utils.flatten_3d_tensor(arr_act), axis=0)
        else:
            mean = np.zeros(d)
        assert mean.shape == (d,)
        arr_centered_arr = arr_act - mean[None, :, None]

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
    def slug(cls) -> str:
        return cls.__name__.lower()
