import typing
import torch

from torch import nn
from torch.utils import hooks

# this is sf

ATTRIBUTE_INTERCEPTED_OUTPUT = "__output"


def attach_hook_intercept_output(
    model: nn.Module, layer: str
) -> typing.Tuple[nn.Module, hooks.RemovableHandle]:
    # remark: this has to be done per architecture
    # Warning: this is only for for ResNet18
    # todo: add `hook`'s returned type

    module = getattr(model, layer)[-1]

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
