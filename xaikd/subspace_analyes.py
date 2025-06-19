from typing import List
from numpy import typing as npt


import torch
from torch import nn

from torch.nn import functional as F
from torch.utils.data import DataLoader

import pandas as pd

from xaikd.metrics import MetricFunction
from xaikd import interceptor

from tqdm.autonotebook import tqdm


def fh_constructor(mean: torch.Tensor, mat_proj: torch.Tensor):

    def fh(module, inp, outp):
        (outp,) = outp
        assert len(outp.shape) == 4, "we assume that the feature map has 4 axes"

        outp = outp - mean
        outp = F.conv2d(outp, mat_proj)
        outp = outp + mean
        return outp

    return fh


@torch.no_grad()
def evaluate_low_rank_approximation(
    model: nn.Module,
    layer: str,
    dataloader: DataLoader,
    metric: MetricFunction,
    arr_ks: List[int],
    mean: npt.NDArray,
    U: npt.NDArray,
    verbose=False,
    device="cpu",
):

    pt_U = torch.from_numpy(U).float().to(device)
    pt_mean = torch.from_numpy(mean).float().to(device)

    module = interceptor.get_module(model, layer)

    arr_result = []

    metric_name = metric._metric_names()
    ref_result = dict(
        zip(
            list(map(lambda n: f"ref_{n}", metric_name)),
            metric(model=model, dataloader=dataloader, verbose=verbose, device=device),
        )
    )

    for k in tqdm(arr_ks, tqdm="evaluating low-rank approximation"):

        pt_Uk = pt_U[:, :k]
        mat_proj = pt_Uk.T @ pt_Uk
        forward_hook = fh_constructor(mean=pt_mean, mat_proj=mat_proj)

        hook = None

        try:
            hook = module.register_forward_hook(forward_hook)
            result = metric(
                model=model, dataloader=dataloader, verbose=verbose, device=device
            )
            assert len(result) == len(metric._metric_names())
            dict_result = dict(zip(metric._metric_names(), result), **ref_result)
            dict_result["k"] = k
            arr_result.append(arr_result)
        finally:
            if hook is not None:
                hook.remove()

    return pd.DataFrame(arr_result)
