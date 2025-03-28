import pytest
import numpy as np
import torch

import tempfile
from pathlib import Path


from xaikd import distillation_policies, bases
from xaikd import utils as putils
from xaikd import bases

from torch import nn

from torch.nn import functional as F


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
@pytest.mark.parametrize(
    "teacher_hw,student_hw",
    [
        ((10, 10), (5, 5)),
        ((5, 5), (10, 10)),
        ((7, 7), (7, 7)),
        ((10, 1), (10, 1)),  # ViT
    ],
)
def test_policy_when_spatial_dimensions_different(
    teacher_dims, student_dims, teacher_hw, student_hw
):
    batch_size = 10
    device = "cpu"
    kwargs = dict(teacher_dims=teacher_dims, student_dims=student_dims, device=device)

    policy = distillation_policies.get_policy("vid", **kwargs)

    teacher_feats = torch.randn(batch_size, teacher_dims, *teacher_hw)
    student_feats = torch.randn(batch_size, student_dims, *student_hw)

    try:
        output = policy(teacher_feats, student_feats)
        assert output is not None
        assert True
    except:
        raise
        assert False, "some exception occurs!"


@pytest.mark.parametrize("teacher_dims", [10])
@pytest.mark.parametrize("student_dims", [6])
@pytest.mark.parametrize(
    "policy",
    [
        # "fitnet-relu", "fitnet-noact",
        "vid",
        "attention-transfer",
        "spkd",
        "vkd",
    ],
)
def test_baseline_policy_callable(teacher_dims, student_dims, policy):
    batch_size = 10
    kwargs = dict(
        teacher_dims=teacher_dims,
        student_dims=student_dims,
        device="cpu",
    )

    policy = distillation_policies.get_policy(policy, **kwargs)

    teacher_feats = torch.randn(batch_size, teacher_dims, 10, 10)
    student_feats = torch.randn(batch_size, student_dims, 5, 5)

    try:
        output = policy(teacher_feats, student_feats)
        assert not torch.isnan(output)
    except:
        raise
        assert False, "some exception occurs!"


@pytest.mark.parametrize("teacher_dims,student_dims", [(10, 5), (20, 2)])
@pytest.mark.parametrize(
    "basis_name",
    [
        "pca",
        "prcaposdef",
        "prcaposdef-entropy0.95",
    ],
)
@pytest.mark.parametrize(
    "parameterization", ["basis-bn-max-normalized", "basis-bn-sum-normalized"]
)
def test_our_policies_callable(
    parameterization, basis_name, teacher_dims, student_dims
):
    rng = np.random.default_rng(seed=1)
    batch_size = 8
    device = "cpu"

    with tempfile.TemporaryDirectory() as tmpdirname:
        act = rng.random((batch_size, teacher_dims, 5)) + 2
        mean_act = putils.flatten_3d_tensor(act).mean(axis=0)
        act -= mean_act[None, :, None]
        logodd = 2 * rng.random((batch_size,)) - 1
        basis = bases.get_basis(basis_name)

        basis.fit(
            arr_act=act,
            arr_ctx=act,
            mean_act=mean_act,
            arr_logodd=logodd,
            logodd_threshold=0,
            seed=1,
            strict_mode=True,
        )

        kwargs = dict(
            teacher_dims=teacher_dims,
            student_dims=student_dims,
            device=device,
            basis=basis,
        )

        policy = distillation_policies.get_policy(parameterization, **kwargs)

        # check that this is callable
        policy(
            torch.randn(5, teacher_dims, 5, 5),
            torch.randn(5, student_dims, 5, 5),
        )

        # from batchnorm
        expected_num_learnable_params = 2 * student_dims

        _, actual_num_learnable_params = putils.count_params_in_list_params(
            policy.parameters()
        )

        np.testing.assert_allclose(
            actual_num_learnable_params, expected_num_learnable_params
        )

        assert True


@pytest.mark.parametrize("teacher_dims,student_dims", [(10, 5), (20, 2)])
@pytest.mark.parametrize(
    "basis_name",
    [
        "pca",
        "prcaposdef",
        "prcaposdef-entropy0.95",
    ],
)
def test_basis_bn_sum_normalized_learnable(basis_name, teacher_dims, student_dims):
    rng = np.random.default_rng(seed=1)
    batch_size = 8
    device = "cpu"

    with tempfile.TemporaryDirectory() as tmpdirname:
        act = rng.random((batch_size, teacher_dims, 5)) + 4
        mean_act = putils.flatten_3d_tensor(act).mean(axis=0)
        act -= mean_act[None, :, None]
        logodd = 2 * rng.random((batch_size,)) - 1
        basis = bases.get_basis(basis_name)

        basis.fit(
            arr_act=act,
            arr_ctx=act,
            mean_act=mean_act,
            arr_logodd=logodd,
            logodd_threshold=0,
            seed=1,
            strict_mode=True,
        )

        kwargs = dict(
            teacher_dims=teacher_dims,
            student_dims=student_dims,
            device=device,
            basis=basis,
        )

        policy = distillation_policies.get_policy(
            "basis-bn-sum-normalized-learnable", **kwargs
        )

        # check that this is callable
        policy(
            torch.randn(5, teacher_dims, 5, 5),
            torch.randn(5, student_dims, 5, 5),
        )

        expected_num_learnable_params = teacher_dims if basis.centering else 0

        # from the weight matrix
        expected_num_learnable_params += teacher_dims * student_dims

        # from batchnorm
        expected_num_learnable_params += 2 * student_dims

        _, actual_num_learnable_params = putils.count_params_in_list_params(
            policy.parameters()
        )

        np.testing.assert_allclose(
            actual_num_learnable_params, expected_num_learnable_params
        )

        assert True


@pytest.mark.parametrize(
    "last_layer_policy,expected",
    [
        ("last-layer:kd", distillation_policies.KLPolicy),
        ("last-layer:dkd", distillation_policies.DKDPolicy),
    ],
)
def test_last_layer_policy(last_layer_policy, expected):
    last_layer_policy = distillation_policies.get_policy(
        last_layer_policy, device="cpu"
    )
    assert isinstance(last_layer_policy, expected)

    rng = torch.Generator()
    rng.manual_seed(1)

    teacher_logits = torch.rand(size=(2, 10), generator=rng)
    student_logits = torch.rand(size=(2, 10), generator=rng)
    targets = torch.randint(size=(2,), low=0, high=9, generator=rng)

    val = last_layer_policy(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        target=targets,
    )

    assert not torch.isnan(val)
