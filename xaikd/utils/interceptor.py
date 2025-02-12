import typing
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils import hooks

from xaikd import utils
from xaikd.utils.metrics import MetricFunction
from tqdm.autonotebook import tqdm


ATTRIBUTE_INTERCEPTED_OUTPUT = "__output"


def get_module(model: nn.Module, layer_str: str) -> nn.Module:
    arr_level_layers = layer_str.split(".")

    parent_module = model

    for attr_name in arr_level_layers:
        parsed_attr_name = utils.parse_number_if_possible(attr_name)

        if parsed_attr_name is not None:
            assert isinstance(parent_module, nn.Sequential)
            assert isinstance(parsed_attr_name, int)

            parent_module = parent_module[parsed_attr_name]
        else:
            parent_module = getattr(parent_module, attr_name)

    module = parent_module

    return module


def attach_hook_intercept_layer_output(
    model: nn.Module, layer: str, should_retain_grad: bool, detach_output: bool
) -> typing.Tuple[nn.Module, hooks.RemovableHandle]:
    module = get_module(model, layer)

    return attach_hook_intercept_module(
        module, should_retain_grad=should_retain_grad, detach_output=detach_output
    )


def attach_hook_intercept_module(
    module: nn.Module, should_retain_grad: bool, detach_output: bool
) -> typing.Tuple[nn.Module, hooks.RemovableHandle]:
    def fh(mod, input, output):
        assert isinstance(output, torch.Tensor)

        setattr(mod, ATTRIBUTE_INTERCEPTED_OUTPUT, output)
        if should_retain_grad:
            output.retain_grad()

        if detach_output:
            return output.detach()

    hook = module.register_forward_hook(fh)

    return module, hook


def get_output(module: nn.Module) -> torch.Tensor:
    output = getattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT)

    delattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT)

    return output


def forward_and_intercept_intermediate_layers(
    model: nn.Module, inp: torch.Tensor, layers: typing.List[str], detach_output: bool
) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
    # todo: add unit tests
    # - all outputs we get are correct
    arr_hooks: typing.List[hooks.RemovableHandle] = []
    arr_modules = []

    arr_intermediate_feats = []

    try:
        # attach hooks to those layers
        for layer in layers:
            module, hook = attach_hook_intercept_layer_output(
                model=model,
                layer=layer,
                should_retain_grad=False,
                detach_output=detach_output,
            )
            arr_modules.append(module)
            arr_hooks.append(hook)

        # one forward pass
        logits = model(inp)

        # extract output of those layers
        for module in arr_modules:
            arr_intermediate_feats.append(get_output(module))

    finally:
        for module, hook in zip(arr_modules, arr_hooks):
            hook.remove()

            if hasattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT):
                delattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT)

    assert len(layers) == len(arr_intermediate_feats)

    return logits, arr_intermediate_feats


def attach_projection_forward_hook_at_layer_and_evaluate_metrics(
    model: nn.Module,
    layer: str,
    dataloader: DataLoader,
    forward_hook_func: typing.Callable[
        [nn.Module, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    metric: MetricFunction,
    device: str,
    verbose=False,
):
    module = get_module(model=model, layer_str=layer)

    hook = module.register_forward_hook(forward_hook_func)

    try:

        return metric(
            model=model, dataloader=dataloader, device=device, verbose=verbose
        )

    finally:
        if hook is not None:
            hook.remove()
