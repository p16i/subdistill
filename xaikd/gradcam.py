import typing
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional as F


from torchvision.models.resnet import ResNet
from torchvision.models.vgg import VGG
from xaikd import utils


def get_module_before_global_pool(model: nn.Module):

    if hasattr(model, "__layer_before_avgpool"):
        # this is for student model
        return getattr(model, getattr(model, "__layer_before_avgpool"))
    elif isinstance(model, ResNet):
        return model.layer4
    elif isinstance(model, VGG):
        return model.features[30]
    else:
        raise


def compute_cam(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    hook = None

    last_layer: int = getattr(model, "__last_layer").weight.shape[0]

    try:
        module = get_module_before_global_pool(model)

        _, hook = utils.interceptor.attach_hook_intercept_module(
            module, should_retain_grad=True, detach_output=False
        )

        logits = model(x) * F.one_hot(y, num_classes=last_layer)

        logits.sum().backward()

        output = utils.interceptor.get_output(module)
        grad = output.grad

        assert len(grad.shape) == 4
        assert grad.shape == output.shape

        b, d, w, h = output.shape

        gradcam = (output * grad.mean(dim=(2, 3), keepdim=True)).sum(dim=1)

        assert gradcam.shape == (b, w, h)

    finally:
        if hook is not None:
            hook.remove()

    return gradcam
