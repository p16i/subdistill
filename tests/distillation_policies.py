import pytest
import numpy as np
import torch

from xaikd import distillation_policies


@pytest.mark.parametrize(
    "string,expected_teacher_layers,expected_student_layers",
    [
        ("layer1,layer2", ["layer1", "layer2"], ["layer1", "layer2"]),
        ("layer1:layer1*,layer2:layer2*", ["layer1", "layer2"], ["layer1*", "layer2*"]),
    ],
)
def test_parsing_layer_string(string, expected_teacher_layers, expected_student_layers):
    print(string)
    (
        actual_teacher_layers,
        actual_student_layers,
    ) = distillation_policies.parse_layer_string(string)

    np.testing.assert_array_equal(actual_teacher_layers, expected_teacher_layers)

    np.testing.assert_array_equal(actual_student_layers, expected_student_layers)


@pytest.mark.parametrize("teacher_dims", [5, 10])
@pytest.mark.parametrize("student_dims", [5, 10])
def test_policy_when_spatial_dimensions_different(teacher_dims, student_dims):
    batch_size = 10
    device = "cpu"
    kwargs = dict(
        teacher_dims=teacher_dims,
        student_dims=student_dims,
        device=device,
    )

    policy = distillation_policies.get_layer_policy("fitnet", **kwargs)

    teacher_feats = torch.randn(batch_size, teacher_dims, 10, 10)
    student_feats = torch.randn(batch_size, student_dims, 5, 5)

    try:
        policy(teacher_feats, student_feats)
        assert True
    except:
        raise
        assert False, "some exception occurs!"
