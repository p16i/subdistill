import numpy as np
import pytest
import torch

from xaikd import models
from xaikd.models.students import canonize_student_model


@torch.no_grad()
@pytest.mark.parametrize(
    "slug",
    [
        "student-32-24-16-8",
        "student-40-32-24-16",
        "student-48-40-32-24",
    ],
)
def test_student_callable(slug):
    torch.manual_seed(1)
    x = torch.rand((7, 3, 224, 224))
    student = models.get_untrained_model(slug, num_classes=10)

    output = student(x)

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
