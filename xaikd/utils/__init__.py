import os
import typing
import numpy.typing as npt
import subprocess
import re

import json
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torchvision

from pathlib import Path


from . import spurious_feature_generator, pixelflipping, ndarray_sampling, modules


from xaikd import interceptor


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

        arr_act.append(selected_act[np.newaxis, :, :])
        arr_ctx.append(selected_ctx[np.newaxis, :, :])

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    assert arr_act.shape == (bs, nc, np.min([num_locations, total_spatial_locations]))

    assert arr_act.shape == arr_ctx.shape

    return arr_act, arr_ctx


def flatten_3d_tensor(x: npt.NDArray) -> npt.NDArray:
    """_summary_

    Args:
        x (torch.Tensor): _description_

    Returns:
        torch.Tensor: 2d tensor whose len(x.shape) == 2
    """
    bs, nc, num_spatial_locations = x.shape

    x = np.transpose(x, [1, 0, 2])

    x = x.reshape((nc, bs * num_spatial_locations))
    x = x.T

    return x


def count_params_in_model(model: torch.nn.Module) -> typing.Tuple[int, int]:
    return count_params_in_list_params(model.parameters())


def convolve_feature_map_with_linear(
    feature_map: torch.Tensor, linear_layer: nn.Module
):
    assert isinstance(linear_layer, (nn.Linear))

    b, d, h, w = feature_map.shape

    out_dims, in_dims = linear_layer.weight.shape

    assert linear_layer.bias is None and in_dims == d

    feature_map = feature_map.flatten(start_dim=2)
    feature_map = feature_map.permute(0, 2, 1)
    feature_map = feature_map.reshape((b * h * w, in_dims))
    feature_map = linear_layer(feature_map)
    feature_map = feature_map.reshape(b, h * w, out_dims)
    feature_map = feature_map.permute(0, 2, 1)
    feature_map = feature_map.reshape(b, out_dims, h, w)

    return feature_map


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


def compute_log_odd_winning(logits: torch.Tensor) -> torch.Tensor:

    ns, nc = logits.shape

    values, _ = torch.topk(logits, dim=1, k=nc)

    logit_winning = values[:, 0]
    lse_others = torch.logsumexp(values[:, 1:], dim=1)
    log_odd = logit_winning - lse_others

    return log_odd


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
    model: nn.Module, selected_classes: typing.List[int]
):
    assert hasattr(model, "__last_layer")
    layer = getattr(model, "__last_layer")

    assert isinstance(layer, nn.Linear)

    layer.weight = nn.Parameter(layer.weight[selected_classes, :])
    layer.bias = nn.Parameter(layer.bias[selected_classes])


@torch.no_grad()
def get_dimensions_at_layers(
    model: nn.Module, dataloader: DataLoader, layers: typing.List[str], device="cpu"
) -> typing.Dict[str, int]:
    assert not model.training

    hooks = []
    modules = []
    try:
        for layer in layers:
            module, hook = interceptor.attach_hook_intercept_layer_output(
                model, layer, should_retain_grad=False, detach_output=False
            )
            hooks.append(hook)
            modules.append(module)

        x, _ = next(iter(dataloader))
        x = x.to(device)

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


def get_git_hash() -> str:
    # ref: https://stackoverflow.com/a/40170206
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        git_revision = out.strip().decode("ascii")
    except OSError:
        git_revision = "Unknown"

    return git_revision


def parse_number_if_possible(text: str) -> typing.Union[None, int]:
    is_int = re.match(r"-?\d+", text) is not None

    if is_int:
        return int(text)
    else:
        return None


def solve_eigh(
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


def adjust_basis_vectors_to_positive_direction(
    U: npt.NDArray,
    x: npt.NDArray,
    strict=False,
):
    # remark: for operatation with U@U.T, this sign correction cancels out,
    # hence having no effect.

    n, d = x.shape

    is_majority_pos_sign = ((x @ U) > 0).mean(axis=0) > 0.5

    assert is_majority_pos_sign.shape == (d,)

    Up = U @ np.diag(is_majority_pos_sign) + U @ np.diag((is_majority_pos_sign - 1))

    if strict:
        np.testing.assert_allclose(Up.T @ Up, np.eye(d), atol=1e-6)

    return Up
