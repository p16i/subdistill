import typing
import numpy.typing as npt

import json
import torch
from torch.utils.data import DataLoader
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
    return count_params_in_list_params(model.parameters())


def count_params_in_list_params(
    params: typing.Iterable[torch.nn.Parameter],
) -> typing.Tuple[int, int]:
    # ref: https://stackoverflow.com/a/49201237

    total, trainable = 0, 0
    for param in params:
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


def logspace(d: int) -> typing.List[int]:
    total = int(np.ceil(np.log2(d)))

    steps = [1] + np.logspace(1, total, num=total, base=2).astype(int).tolist()

    if steps[-1] > d:
        steps[-1] = d

    return steps


def is_permuation_matrix(x: npt.NDArray) -> bool:
    # ref: https://stackoverflow.com/a/28896366
    return (
        x.ndim == 2
        and x.shape[0] == x.shape[1]
        and (x.sum(axis=0) == 1).all()
        and (x.sum(axis=1) == 1).all()
        and ((x == 1) | (x == 0)).all()
    )


def modify_last_layer_for_subclasses(
    layer: nn.Linear, selected_classes: typing.List[int]
):
    assert isinstance(layer, nn.Linear)

    layer.weight = nn.Parameter(layer.weight[selected_classes, :])
    layer.bias = nn.Parameter(layer.bias[selected_classes])


@torch.no_grad()
def get_dimensions_at_layers(
    model: nn.Module, dataloader: DataLoader, layers: typing.List[str]
) -> typing.Dict[str, int]:
    assert not model.training

    hooks = []
    modules = []
    try:
        for layer in layers:
            module, hook = interceptor.attach_hook_intercept_layer_output(
                model, layer, should_retain_grad=False
            )
            hooks.append(hook)
            modules.append(module)

        x, _ = next(iter(dataloader))

        _ = model(x)

        dimensions = dict()
        for layer, module in zip(layers, modules):
            output = interceptor.get_output(module)
            _, d, _, _ = output.shape
            dimensions[layer] = d

    finally:
        for hook in hooks:
            hook.remove()

    return dimensions
