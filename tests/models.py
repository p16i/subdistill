import torch
import pytest
import numpy as np

from xaikd import models, constants, utils


@pytest.mark.parametrize(
    "slug",
    [
        "cifar100-resnet18-v1",
        "imagenet-resnet18-tv",
        "imagenet-resnet34-tv",
        "imagenet-resnet50-tv",
        "imagenet-resnet101-tv",
        "imagenet-resnet152-tv",
        "imagenet-vgg11-tv",
        "imagenet-vgg16-tv",
        "imagenet-vgg16bn-tv",
    ],
)
@pytest.mark.slow
@torch.no_grad()
def test_get_models(slug):
    torch.manual_seed(1)
    model = models.get_trained_model(slug)
    assert not model.training
    assert model is not None
    assert getattr(model, "__name") == slug
    assert len(getattr(model, "__layer_dimension").keys()) > 0

    dataset = slug.split("-")[0]
    if dataset == "cifar100":
        data = torch.rand(5, 3, 32, 32)
    elif dataset == "imagenet":
        data = torch.rand(5, 3, 224, 224)

    output = model(data)


@pytest.mark.parametrize(
    "slug",
    [
        "cifar100-resnet18-v1",
        "imagenet-resnet18-tv",
        "imagenet-resnet34-tv",
        "imagenet-resnet50-tv",
        "imagenet-resnet101-tv",
        "imagenet-resnet152-tv",
        "imagenet-vgg11-tv",
        "imagenet-vgg16-tv",
        "imagenet-vgg16bn-tv",
    ],
)
@pytest.mark.slow
@torch.no_grad()
def test_verify_layer_dimensions(slug):
    torch.manual_seed(1)
    model = models.get_trained_model(slug)

    device = utils.get_device()

    model.to(device)

    dataset, arch, variant = slug.split("-")
    if dataset == "cifar100":
        data = torch.rand(5, 3, 32, 32)
    elif dataset == "imagenet":
        data = torch.rand(5, 3, 224, 224)

    data.to(device)

    for layer, expected_dims in constants.ARCH_LAYER_DIMENSIONS[arch].items():
        try:
            module, hook = utils.interceptor.attach_hook_intercept_layer_output(
                model, layer, should_retain_grad=False
            )

            _ = model(data)

            act = utils.interceptor.get_output(module)

            _, actual_dims, _, _ = act.shape

            assert actual_dims == expected_dims, f"arch={arch}; layer={layer}"

        finally:
            hook.remove()
