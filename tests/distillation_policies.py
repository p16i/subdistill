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
    kwargs = dict(
        teacher_dims=teacher_dims,
        student_dims=student_dims,
        device=device,
    )

    policy = distillation_policies.get_layer_policy("fitnet-noact", **kwargs)

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
    "policy", ["fitnet-relu", "fitnet-noact", "vid", "attention-transfer"]
)
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
        act = rng.random((batch_size, teacher_dims, 5))
        logodd = 2 * rng.random((batch_size,)) - 1
        basis = bases.get_basis("pca")

        basis.fit(
            arr_act=act, arr_ctx=act, arr_logodd=logodd, logodd_threshold=0, seed=1
        )

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


@pytest.mark.parametrize(
    "last_layer_policy,expected",
    [
        ("kd", distillation_policies.KLPolicy),
        ("dkd", distillation_policies.DKDPolicy),
    ],
)
def test_last_layer_policy(last_layer_policy, expected):
    last_layer_policy = distillation_policies.get_last_layer_policy(last_layer_policy)
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


def test_vkd():
    feat_teacher = torch.randn(20, 30, 7, 7)
    feat_student = torch.randn(20, 10, 7, 7)

    vkd = distillation_policies.VkD(teacher_dims=30, student_dims=10, device="cpu")

    output = vkd(feat_teacher, feat_student)

    assert not torch.isnan(output)


def test_vkd_extended():
    feat_teacher = torch.randn(20, 30, 7, 7)
    feat_student = torch.randn(20, 10, 7, 7)

    vkd = distillation_policies.VkDModified(
        teacher_dims=30, student_dims=10, device="cpu"
    )

    output = vkd(feat_teacher, feat_student)

    assert not torch.isnan(output)


@pytest.mark.skip(reason="obsolete")
def test_basis_rotation():
    basis = bases.get_basis("pca")

    rng = torch.Generator()
    rng.manual_seed(1)

    dims_teacher = 20
    dims_student = 10

    feat_teacher = torch.rand((10, dims_teacher, 4, 4), generator=rng)
    feat_student = torch.rand((10, dims_student, 4, 4), generator=rng)

    arr_act = torch.rand((40, dims_teacher), generator=rng).detach().cpu().numpy()
    arr_logodd = torch.rand((40,), generator=rng).detach().cpu().numpy()

    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_act,
        arr_logodd=arr_logodd,
        logodd_threshold=0.0,
        seed=1,
    )

    policy = distillation_policies.OrthogonalBasisRotationPolicy(
        teacher_dims=dims_teacher, student_dims=dims_student, device="cpu", basis=basis
    )

    num_params, num_trainable_params = putils.count_params_in_model(policy)

    assert num_params == num_trainable_params == (dims_student**2) + 1

    # simulate update

    student_feat_transform = policy.transformer_student_feats
    w_before = student_feat_transform.rotation.weight.detach().numpy()  # type: ignore
    scaling_before = policy.transformer_student_feats.scaling.detach().numpy()  # type: ignore
    np.testing.assert_allclose(scaling_before, 1)

    np.testing.assert_allclose(w_before @ w_before.T, np.eye(dims_student), atol=1e-6)
    np.testing.assert_allclose(w_before.T @ w_before, np.eye(dims_student), atol=1e-6)

    optim = torch.optim.SGD(policy.parameters(), lr=1e-3)

    loss = policy(feat_teacher, feat_student)

    loss.backward()

    optim.step()

    w_after = policy.transformer_student_feats.rotation.weight.detach().numpy()  # type: ignore
    np.testing.assert_allclose(w_after @ w_after.T, np.eye(dims_student), atol=1e-6)
    np.testing.assert_allclose(w_after.T @ w_after, np.eye(dims_student), atol=1e-6)
    scaling_after = policy.transformer_student_feats.scaling.detach().numpy()  # type: ignore

    assert scaling_after != 1
