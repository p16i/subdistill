import numpy as np
import pytest
import torch
import torch.nn as nn

from xaikd import models, constants, utils
from xaikd.models.students import canonize_student_model


@pytest.mark.skip(reason="This test is obsolete")
@pytest.mark.parametrize("arr_dims", constants.ARR_STUDENT_DIMENSIONS)
def test_student_callable(arr_dims):
    torch.manual_seed(1)
    x = torch.rand((7, 3, 224, 224))
    slug = "-".join(np.array(arr_dims).astype(str))
    student = models.get_untrained_model(f"student-{slug}", num_classes=10)

    assert student.training

    output = student(x)
    output.sum().backward()

    assert output.shape == (7, 10)


@pytest.mark.skip(reason="This test is obsolete")
@torch.no_grad()
def test_canonize_student():
    torch.manual_seed(1)
    x_train = torch.rand(5, 3, 224, 224)
    num_classes = 6

    model = models.get_untrained_model("student-32-24-16-8", num_classes=num_classes)
    model(x_train)

    model.eval()

    canonized_model = canonize_student_model(model)

    canonized_model.eval()

    x = torch.rand(5, 3, 224, 224)

    expected = model(x)
    actual = canonized_model(x)

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize(
    "student_name",
    [
        "student-cifar-resnet18-dims16-16-8-8",
        "student-cifar-resnet18-dims32-32-16-16",
        "student-cifar-resnet18-dims64-64-32-32",
    ],
)
@pytest.mark.slow()
def test_cifar100_resnet(student_name):
    x = torch.rand(5, 3, 32, 32)
    model = models.get_untrained_model(student_name, num_classes=10)
    assert isinstance(model, models.resnet.resnet.ResNet)
    assert isinstance(model.maxpool, nn.Identity)
    model.eval()
    output = model(x)
    assert torch.isfinite(output).all()
    assert output.shape == (5, 10)


@torch.no_grad()
@pytest.mark.parametrize(
    "student_name",
    [
        "student-resnet18-dims16-16-8-8",
        "student-resnet18-dims32-32-16-16",
        "student-resnet18-dims64-64-32-32",
    ],
)
@pytest.mark.slow()
def test_resnet(student_name):
    x = torch.rand(5, 3, 224, 224)
    model = models.get_untrained_model(student_name, num_classes=10)
    assert isinstance(model, models.resnet.resnet.ResNet)
    assert isinstance(model.maxpool, nn.MaxPool2d)
    model.eval()
    output = model(x)
    assert torch.isfinite(output).all()
    assert output.shape == (5, 10)


@torch.no_grad()
@pytest.mark.parametrize(
    "student_name",
    [
        "student-resnet18-2blocks-dims54-16-8",
        "student-resnet18-2blocks-dims64-32-16",
        "student-resnet18-2blocks-dims128-64-32",
    ],
)
@pytest.mark.slow()
def test_resnet_2l(student_name):
    x = torch.rand(5, 3, 224, 224)
    output_size = int(student_name.split("-dims")[-1].split("-")[0])
    model = models.get_untrained_model(student_name, num_classes=10)
    # assert isinstance(model, models.resnet.resnet.ResNet)
    assert isinstance(model.stem, nn.AdaptiveAvgPool2d)
    assert model.stem.output_size == output_size

    model.eval()
    output = model(x)
    assert torch.isfinite(output).all()
    assert output.shape == (5, 10)


@pytest.mark.parametrize(
    "student_name",
    [
        "student-resnet18-transferred2layers-dims32-16",
        "student-resnet18-transferred2layers-dims64-32",
    ],
)
@torch.no_grad()
def test_resnet_transferred_2layers(student_name):
    orig_model = models.get_trained_model("celeba-resnet18-finetunedv1")

    # Test that the transf""" er works for a 2-layer ResNet
    x = torch.rand(5, 3, 224, 224)

    model = models.get_untrained_model(student_name, num_classes=10)

    # count number params

    n_params, n_trainable_params = utils.count_params_in_model(model)

    layers = [model.layer3, model.layer4, model.fc]
    expected_trainable_params = sum(
        utils.count_params_in_model(layer)[1] for layer in layers
    )

    # fixme: check forword pass the same.

    np.testing.assert_equal(
        n_trainable_params,
        expected_trainable_params,
    )

    _, arr_ori_output = utils.interceptor.forward_and_intercept_intermediate_layers(
        orig_model, x, layers=["layer1", "layer2"], detach_output=False
    )

    _, arr_student_output = utils.interceptor.forward_and_intercept_intermediate_layers(
        model, x, layers=["layer1", "layer2"], detach_output=False
    )

    for actual_output, expected_output in zip(arr_student_output, arr_ori_output):
        np.testing.assert_allclose(actual_output, expected_output, atol=1e-6)

    # assert isinstance(model, models.resnet.resnet.ResNet)
    # assert isinstance(model.stem, nn.AdaptiveAvgPool2d)

    # model.eval()
    # output = model(x)
    # assert torch.isfinite(output).all()
    # assert output.shape == (5, 10) """
