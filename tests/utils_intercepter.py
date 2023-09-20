import pytest
from types import MethodType

import torch

from xaikd import models, utils
from xaikd.utils import interceptor


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


@pytest.mark.parametrize("slug", ("cifar10-resnet18-p1", "cifar100-resnet18-p1"))
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
            model1, layer, should_retain_grad=False
        )

        logits = model1(dummy_input)
        output = getattr(module, "__output")
        delattr(module, "__output")

        assert torch.allclose(output, layers_output[int(layer[-1]) - 1])

        assert torch.allclose(logits, expected_logit)

    finally:
        hook.remove()


@pytest.mark.parametrize("slug", ("resnet18cifarcompr2",))
@pytest.mark.parametrize("layer", ("layer1", "layer2", "layer3", "layer4"))
# @pytest.mark.slow()
def test_student_extra_interception(slug, layer):
    model1 = models.get_untrained_model(slug, num_classes=5)

    try:
        module, hook = interceptor.attach_hook_intercept_layer_output(
            model1, layer, should_retain_grad=False
        )

        assert isinstance(module, torch.nn.BatchNorm2d)

    finally:
        hook.remove()
