import numpy as np
import pytest
import torch

from xaikd import models, constants
from xaikd.models.students import canonize_student_model


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
        "student-cifar-resnet18-16",
        "student-cifar-resnet18-32",
        "student-cifar-resnet18-64",
    ],
)
@pytest.mark.slow()
def test_cifar100_resnet(student_name):
    x = torch.rand(5, 3, 32, 32)
    model = models.get_untrained_model(student_name, num_classes=10)
    model.eval()
    output = model(x)
    assert torch.isfinite(output).all()
    assert output.shape == (5, 10)


@torch.no_grad()
@pytest.mark.parametrize(
    "student_name",
    [
        "student-resnet18-16",
        "student-resnet18-32",
        "student-resnet18-64",
        "student-resnet18-d56-56-40-40",
    ],
)
@pytest.mark.slow()
def test_resnet(student_name):
    x = torch.rand(5, 3, 224, 224)
    model = models.get_untrained_model(student_name, num_classes=10)
    model.eval()
    output = model(x)
    assert torch.isfinite(output).all()
    assert output.shape == (5, 10)
