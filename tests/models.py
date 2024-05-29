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
        "imagenet-vgg11bn-tv",
        "imagenet-vgg13-tv",
        "imagenet-vgg13bn-tv",
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
    assert isinstance(getattr(model, "__last_layer"), torch.nn.Module)

    device = utils.get_device()

    model = model.to(device)

    dataset, arch, variant = slug.split("-")
    if dataset == "cifar100":
        data = torch.rand(5, 3, 32, 32)
    elif dataset == "imagenet":
        data = torch.rand(5, 3, 224, 224)

    data = data.to(device)

    # model is forwardable
    _ = model(data)

    # verify that the dimension mapping is correct
    for layer, expected_dims in constants.ARCH_LAYER_DIMENSIONS[arch].items():
        try:
            module, hook = utils.interceptor.attach_hook_intercept_layer_output(
                model, layer, should_retain_grad=False, detach_output=False
            )

            _ = model(data)

            act = utils.interceptor.get_output(module)

            _, actual_dims, _, _ = act.shape

            assert actual_dims == expected_dims, f"arch={arch}; layer={layer}"

        finally:
            hook.remove()

    # verify that modify output work
    with torch.no_grad():
        utils.modify_last_layer_for_subclasses(model, list(range(8)))
        output = model(data).cpu().numpy()
        assert output.shape == (5, 8)


def _test_split_model(slug, layer, split_func):
    device = utils.get_device()
    model = models.get_trained_model(slug)
    model.to(device)

    head, classifier = split_func(model, layer)
    if "imagenet" in slug:
        input = torch.randn(10, 3, 224, 224)
    else:
        input = torch.randn(10, 3, 32, 32)

    input = input.to(device)

    with torch.no_grad():
        actual = classifier(head(input)).cpu().numpy()
        expected = model(input).cpu().numpy()

        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "model,layer",
    [
        ("imagenet-vgg16-tv", "features.23"),
        ("imagenet-resnet18-tv", "layer2"),
    ],
)
def test_split_model(model, layer):
    _test_split_model(model, layer, models.split_model_at_layer)
