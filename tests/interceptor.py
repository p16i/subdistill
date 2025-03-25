import pytest
from types import MethodType

import torch
from torch import nn

from xaikd import models, utils, interceptor

from collections import OrderedDict

from copy import deepcopy
import numpy as np


DEVICE = utils.get_device()


def test_intercept_intermediate_outputs():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Conv2d(3, 8, 3, 1)
            self.act1 = nn.ReLU()
            self.lin2 = nn.Conv2d(8, 16, 2, 1)
            self.act2 = nn.ReLU()
            self.lin3 = nn.Conv2d(16, 32, 3, 1)
            self.act3 = nn.ReLU()
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            a1 = self.act1(self.lin1(x))

            a2 = self.act2(self.lin2(a1))

            a3 = self.act3(self.lin3(a2))

            out = self.avgpool(a3)

            out = torch.flatten(out, 1)

            out = self.fc(out)
            return out, (a1, a2, a3)

    trng = torch.Generator()
    trng.manual_seed(1)

    x = torch.randn(6, 3, 28, 28, generator=trng)

    ref_model = DummyModel()
    model = deepcopy(ref_model)

    expected_output, expected_intermediate_outputs = ref_model(x)

    actual_output, actual_intermediate_outputs = (
        utils.interceptor.forward_and_intercept_intermediate_layers(
            model=model, inp=x, layers=["act1", "act2", "act3"], detach_output=False
        )
    )

    np.testing.assert_allclose(
        actual_output[0].detach().numpy(), expected_output.detach().numpy()
    )

    for actual, expected in zip(
        actual_intermediate_outputs, expected_intermediate_outputs
    ):
        np.testing.assert_allclose(actual.detach().numpy(), expected.detach().numpy())


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


# fixme: test nfnet interceptors


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
