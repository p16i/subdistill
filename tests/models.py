import torch
import pytest
import numpy as np

from xaikd import models, constants, utils


@torch.no_grad()
def _test_get_model(slug):
    torch.manual_seed(1)
    model = models.get_trained_model(slug)
    assert not model.training
    assert model is not None
    assert getattr(model, "__name") == slug
    assert isinstance(getattr(model, "__last_layer"), torch.nn.Module)

    device = utils.get_device()

    model = model.to(device)

    dataset, arch, variant = slug.split("-")
    if dataset == "cifar100":
        data = torch.rand(5, 3, 32, 32)
    elif dataset in ["imagenet", "celeba"]:
        data = torch.rand(5, 3, 224, 224)

    data = data.to(device)

    # model is forwardable
    _ = model(data)

    # verify that modify output work
    # todo: this should be with utils.tests
    with torch.no_grad():
        utils.modify_last_layer_for_subclasses(model, list(range(8)))
        output = model(data).cpu().numpy()
        assert output.shape == (5, 8)


@torch.no_grad()
def _test_split_model(slug, layer, split_func, atol=1e-6):
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

        np.testing.assert_allclose(actual, expected, atol=atol)


@pytest.mark.parametrize(
    "model,arr_layers",
    [
        ("imagenet-vitb-tv", "encoder.layers.8,encoder.layers.11"),
        ("imagenet-resnet18-tv", "layer3,layer4"),
        ("imagenet-resnet50-tv", "layer3,layer4"),
        ("imagenet-vgg16-tv", "features.23,features.30"),
        ("imagenet-nfnetf0-dm", "stages.2,stages.3"),
    ],
)
def test_split_models_callable(model, arr_layers):

    model = models.get_trained_model(model)
    for layer in arr_layers.split(","):
        models.split_model_at_layer(model, layer)
