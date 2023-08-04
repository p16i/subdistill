import typing
import numpy.typing as npt

import json
import torch
from torch import nn
import numpy as np

from pathlib import Path


from . import interceptor


T = typing.TypeVar("T")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def _string_serializer(item):
    # return f"{item}"
    if isinstance(item, object):
        if hasattr(item, "__name"):
            return getattr(item, "__name")
        else:
            return f"{item}"
    else:
        return item


def dump_json_with_string_serializer(dest: Path, data: dict):
    with open(dest, "w") as fh:
        json.dump(data, fh, indent=4, sort_keys=True, default=_string_serializer)


def dump_json(dest: Path, data: dict):
    with open(dest, "w") as fh:
        json.dump(data, fh, indent=4, sort_keys=True)


def subsample_tensors(
    act: npt.NDArray,
    ctx: npt.NDArray,
    num_locations=20,
    rng=np.random.default_rng(),
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    assert len(act.shape) == 4

    bs, nc, h, w = act.shape

    total_spatial_locations = w * h
    arr_act = []
    arr_ctx = []

    for ix in range(bs):
        _a = act[ix]
        _c = ctx[ix]

        assert _a.shape == (nc, h, w)

        selected = rng.permutation(total_spatial_locations)[:num_locations]
        flattened_act = _a.reshape((nc, -1))
        flattened_ctx = _c.reshape((nc, -1))
        selected_act = flattened_act[:, selected]
        selected_ctx = flattened_ctx[:, selected]

        arr_act.append(selected_act.T)
        arr_ctx.append(selected_ctx.T)

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    assert arr_act.shape == (bs * np.min([num_locations, total_spatial_locations]), nc)

    return arr_act, arr_ctx


def count_params_in_model(model: torch.nn.Module) -> typing.Tuple[int, int]:
    # ref: https://stackoverflow.com/a/49201237

    total, trainable = 0, 0
    for param in model.parameters():
        n = param.numel()
        total += n

        if param.requires_grad:
            trainable += n

    return total, trainable


def deactivate_requires_grad(model: torch.nn.Module):
    # remark: https://github.com/lightly-ai/lightly/blob/master/lightly/models/utils.py#L166
    for param in model.parameters():
        param.requires_grad = False


def freeze_model(model: torch.nn.Module) -> torch.nn.Module:
    deactivate_requires_grad(model)
    model.eval()

    return model


def query_module_children_with_type(
    module: nn.Module, module_type: typing.Type[T]
) -> typing.List[T]:
    basket = []
    for child in module.children():
        if isinstance(child, module_type):
            basket.append(child)
        else:
            basket.extend(query_module_children_with_type(child, module_type))

    return basket
