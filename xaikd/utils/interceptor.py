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
    model: nn.Module, layer: str
) -> typing.Tuple[nn.Module, hooks.RemovableHandle]:
    # remark: this has to be done per architecture
    # Warning: this is only for for ResNet18
    # todo: add `hook`'s returned type
    module = get_module(model, layer)

    return attach_hook_intercept_module(module)


def attach_hook_intercept_module(
    module: nn.Module,
) -> typing.Tuple[nn.Module, hooks.RemovableHandle]:
    def fh(mod, input, output):
        assert isinstance(output, torch.Tensor)

        setattr(mod, ATTRIBUTE_INTERCEPTED_OUTPUT, output)
        output.retain_grad()

    hook = module.register_forward_hook(fh)

    return module, hook


def get_output(module: nn.Module) -> torch.Tensor:
    output = getattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT)

    delattr(module, ATTRIBUTE_INTERCEPTED_OUTPUT)

    return output
