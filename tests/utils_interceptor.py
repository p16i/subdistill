import pytest
from types import MethodType

import torch
from torch import nn

from xaikd import models, utils
from xaikd.utils import interceptor

from collections import OrderedDict

import numpy as np


DEVICE = utils.get_device()


# resnet18
def _overriden_resnet18_forward_impl(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)

    x = self.avgpool(out4)
    x = torch.flatten(x, 1)
    x = self.fc(x)

    return x, (out1, out2, out3, out4)


@pytest.mark.parametrize("slug", ("cifar100-resnet18-v1",))
@pytest.mark.parametrize("layer", ("layer1", "layer2", "layer3", "layer4"))
@pytest.mark.slow()
def test_resnet_layer_interception(slug, layer):
    print(f"Testing on {DEVICE}")

    model1 = models.get_trained_model(slug).to(DEVICE)
    model2 = models.get_trained_model(slug).to(DEVICE)

    dummy_input = torch.randn((10, 3, 32, 32)).to(DEVICE)

    model2._forward_impl = MethodType(_overriden_resnet18_forward_impl, model2)

    expected_logit, layers_output = model2(dummy_input)

    try:
        module, hook = interceptor.attach_hook_intercept_layer_output(
            model1, layer, should_retain_grad=False, detach_output=False
        )

        logits = model1(dummy_input)
        output = getattr(module, "__output")
        delattr(module, "__output")

        assert torch.allclose(output, layers_output[int(layer[-1]) - 1])

        assert torch.allclose(logits, expected_logit)

    finally:
        hook.remove()


@pytest.mark.parametrize(
    "model_name,layers,input_size",
    [
        # ("cifar100-vgg11-v1", ("features.10", "features.15", "features.20"), (32, 32)),
        ("imagenet-vgg16-tv", ("features.9", "features.16"), (224, 224)),
    ],
)
@pytest.mark.slow()
def test_vgg_layer_interception(model_name, layers, input_size):
    print(f"Testing on {DEVICE}")

    for layer in layers:
        print(layer)
        layer_index = int(layer.split(".")[1])

        model1 = models.get_trained_model(model_name).to(DEVICE)
        model2 = models.get_trained_model(model_name).to(DEVICE)

        dummy_input = torch.randn((5, 3, *input_size)).to(DEVICE)

        model2_first_path = model2.features[: layer_index + 1]

        expected_output = model2_first_path(dummy_input)
        expected_logit = model2(dummy_input)

        try:
            module, hook = interceptor.attach_hook_intercept_layer_output(
                model1, layer, should_retain_grad=False, detach_output=False
            )

            assert isinstance(module, torch.nn.MaxPool2d)

            logits = model1(dummy_input)
            output = getattr(module, "__output")
            delattr(module, "__output")

            assert torch.allclose(output, expected_output)

            assert torch.allclose(logits, expected_logit)

        finally:
            hook.remove()


@pytest.mark.parametrize(
    "model_name,layers,expected_parameterization_module",
    [
        (
            "resnet18xscifarcompr1",
            ("layer1", "layer2", "layer3", "layer4"),
            torch.nn.BatchNorm2d,
        ),
        (
            "resnet18xscifarcompr1lin",
            ("layer1", "layer2", "layer3", "layer4"),
            torch.nn.Conv2d,
        ),
        (
            "resnet18xscifarcompr1diag",
            ("layer1", "layer2", "layer3", "layer4"),
            models.resnet.DiagonalScaling,
        ),
        (
            "resnet18xscifarcompr1",
            ("layer1", "layer2", "layer3", "layer4"),
            torch.nn.BatchNorm2d,
        ),
        ("vgg8xs", ("features.8",), torch.nn.BatchNorm2d),
    ],
)
def test_student_extra_interception(
    model_name, layers, expected_parameterization_module
):
    model1 = models.get_untrained_model(model_name, num_classes=5)

    for layer in layers:
        try:
            module, hook = interceptor.attach_hook_intercept_layer_output(
                model1, layer, should_retain_grad=False, detach_output=False
            )

            assert isinstance(module, expected_parameterization_module)

        finally:
            hook.remove()


@pytest.mark.parametrize("detach_output", [True, False])
def test_forward_hook_partition_parameter_update(detach_output):
    torch.manual_seed(1)

    x = torch.randn(10, 2)

    model = nn.Sequential(
        OrderedDict(
            [
                ("layer1", nn.Linear(in_features=2, out_features=3)),
                ("layer2", nn.Linear(in_features=3, out_features=10)),
            ]
        )
    )

    try:
        _, hook = interceptor.attach_hook_intercept_layer_output(
            model, "layer1", should_retain_grad=False, detach_output=detach_output
        )
        loss = model(x).sum()
        loss.backward()
    finally:
        hook.remove()

    assert not model.layer2.weight.grad is None
    assert not model.layer2.bias.grad is None

    if detach_output:
        assert model.layer1.weight.grad is None
        assert model.layer1.bias.grad is None
    else:
        assert not model.layer1.weight.grad is None
        assert not model.layer1.bias.grad is None


@pytest.mark.parametrize("detach_output", [True, False])
def test_forward_and_intercept_withpartition_parameter_update(detach_output):
    torch.manual_seed(1)

    x = torch.randn(10, 2)

    model = nn.Sequential(
        OrderedDict(
            [
                ("layer1", nn.Linear(in_features=2, out_features=3)),
                ("layer2", nn.Linear(in_features=3, out_features=10)),
            ]
        )
    )

    actual_output, (actual_activation,) = (
        interceptor.forward_and_intercept_intermediate_layers(
            model=model, inp=x, layers=["layer1"], detach_output=detach_output
        )
    )

    loss = actual_output.sum()
    loss.backward()

    with torch.no_grad():
        np.testing.assert_allclose(
            actual_output.detach().numpy(), model(x).detach().numpy()
        )
        np.testing.assert_allclose(
            actual_activation.detach().numpy(), model.layer1(x).detach().numpy()
        )

    assert not model.layer2.weight.grad is None
    assert not model.layer2.bias.grad is None

    if detach_output:
        assert model.layer1.weight.grad is None
        assert model.layer1.bias.grad is None
    else:
        assert not model.layer1.weight.grad is None
        assert not model.layer1.bias.grad is None
