import pytest

import torch

from xaikd import models, gradcam, utils


@pytest.mark.parametrize(
    "model_name",
    [
        "imagenet-vgg16-tv",
        "imagenet-resnet18-tv",
    ],
)
def test_grad_cam_trained_model(model_name):

    device = utils.get_device()

    model = models.get_trained_model(model_name)
    model.to(device)

    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224).to(device)
        y = torch.tensor([100]).to(device)

    cam = gradcam.compute_cam(model=model, x=x, y=y).detach().cpu()

    assert not torch.isnan(cam).any()


@pytest.mark.parametrize(
    "model_name",
    ["vggcustomimagenetdims-32-24-24-10"],
)
def test_grad_cam_untrained_model(model_name):

    model = models.get_untrained_model(model_name, num_classes=10)

    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        y = torch.tensor([8])

    cam = gradcam.compute_cam(model=model, x=x, y=y)

    assert not torch.isnan(cam).any()
