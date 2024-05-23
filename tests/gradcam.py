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
@pytest.mark.parametrize("freeze_model", [False, True])
def test_grad_cam_trained_model(model_name, freeze_model):

    if freeze_model:
        model = utils.freeze_model(models.get_trained_model(model_name))
    else:
        model = models.get_trained_model(model_name)

    x = torch.randn(1, 3, 224, 224)
    y = torch.tensor([100])

    cam = gradcam.compute_cam(model=model, x=x, y=y)

    assert not torch.isnan(cam).any()


@pytest.mark.parametrize(
    "model_name",
    ["vggcustomimagenetdims-32-24-24-10"],
)
def test_grad_cam_untrained_model(model_name):

    model = models.get_untrained_model(model_name, num_classes=10)

    x = torch.randn(1, 3, 224, 224)
    y = torch.tensor([8])
    model(x)

    cam = gradcam.compute_cam(model=model, x=x, y=y)

    assert not torch.isnan(cam).any()
