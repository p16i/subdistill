import pytest
import numpy as np
import torch

import tempfile
from pathlib import Path

from xaikd import distillation_policies, bases
from xaikd import utils as putils


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


@pytest.mark.parametrize("teacher_dims", [10])
@pytest.mark.parametrize("student_dims", [6])
@pytest.mark.parametrize("policy", ["fitnet", "fitnet-1l", "vid", "attention-transfer"])
def test_baseline_policy_callable(teacher_dims, student_dims, policy):
    batch_size = 10
    device = "cpu"
    kwargs = dict(
        teacher_dims=teacher_dims,
        student_dims=student_dims,
        device=device,
    )

    policy = distillation_policies.get_layer_policy(policy, **kwargs)

    teacher_feats = torch.randn(batch_size, teacher_dims, 10, 10)
    student_feats = torch.randn(batch_size, student_dims, 5, 5)

    try:
        policy(teacher_feats, student_feats)
        assert True
    except:
        raise
        assert False, "some exception occurs!"


@pytest.mark.parametrize("teacher_dims,student_dims", [(10, 5), (20, 2)])
def test_basis_identity_learnable(teacher_dims, student_dims):
    rng = np.random.default_rng(seed=1)
    batch_size = 8
    device = "cpu"

    with tempfile.TemporaryDirectory() as tmpdirname:
        output_dir = Path(tmpdirname)
        act = rng.random((batch_size, teacher_dims))
        print(act.shape)
        basis = bases.get_basis("random--uncentered")

        basis.fit(arr_act=act, arr_ctx=act, seed=1)

        kwargs = dict(
            teacher_dims=teacher_dims,
            student_dims=student_dims,
            device=device,
            basis=basis,
        )

        policy = distillation_policies.get_layer_policy(
            "basis-identity-learnable", **kwargs
        )

        expected_num_learnable_params = teacher_dims * student_dims

        _, actual_num_learnable_params = putils.count_params_in_list_params(
            policy.parameters()
        )

        np.testing.assert_allclose(
            actual_num_learnable_params, expected_num_learnable_params
        )
