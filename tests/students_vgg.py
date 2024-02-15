import numpy as np
import pytest
import torch

from xaikd import models
from xaikd.models.vgg import canonize_model


@torch.no_grad()
def test():
    torch.manual_seed(1)
    x = torch.rand((7, 3, 224, 224))
    student_vgg = models.get_untrained_model(
        "vggcustomimagenetdims-32-24-24-10", num_classes=10
    )

    output = student_vgg(x)
    print("output.shape", output.shape)

    assert output.shape == (7, 10)


@torch.no_grad()
def test_canonize_vgg_student():
    torch.manual_seed(1)
    x_train = torch.rand(5, 3, 224, 224)
    num_classes = 6

    model = models.get_untrained_model(
        "vggcustomimagenetdims-32-24-24-10", num_classes=num_classes
    )
    model(x_train)

    model.eval()

    canonized_model = canonize_model(model)

    canonized_model.eval()

    x = torch.rand(5, 3, 224, 224)

    expected = model(x)
    actual = canonized_model(x)

    np.testing.assert_allclose(actual, expected, atol=1e-6)
