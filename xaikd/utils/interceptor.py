import typing
import torch

from torch import nn
from torch.utils import hooks

# this is sf

ATTRIBUTE_INTERCEPTED_OUTPUT = "__output"


def get_module(model: nn.Module, layer_str: str) -> nn.Module:
    slugs = layer_str.split(".")

    if len(slugs) == 1:
        module = getattr(model, layer_str)[-1]
    elif len(slugs) == 2:
        layer, index = slugs
        module = getattr(model, layer)[int(index)]
    else:
        raise ValueError(f"layer={layer_str}; not exists")

    return module


def attach_hook_intercept_layer_output(
    model: nn.Module, layer: str, should_retain_grad: bool
) -> typing.Tuple[nn.Module, hooks.RemovableHandle]:
    # remark: this has to be done per architecture
    # Warning: this is only for for ResNet18
    # todo: add `hook`'s returned type
    module = get_module(model, layer)

    return attach_hook_intercept_module(module, should_retain_grad=should_retain_grad)


def attach_hook_intercept_module(
    module: nn.Module, should_retain_grad: bool
) -> typing.Tuple[nn.Module, hooks.RemovableHandle]:
    def fh(mod, input, output):
        assert isinstance(output, torch.Tensor)

        setattr(mod, ATTRIBUTE_INTERCEPTED_OUTPUT, output)
        if should_retain_grad:
            output.retain_grad()

    hook = module.register_forward_hook(fh)

    return module, hook


def get_output(module: nn.Module) -> torch.Tensor:
    output = getattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT)

    delattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT)

    return output


def forward_and_intercept_intermediate_layers(
    model: nn.Module, inp: torch.Tensor, layers: typing.List[str]
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
                model=model, layer=layer, should_retain_grad=False
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
