import pytest
import numpy as np
import torch

import tempfile
from pathlib import Path


from copy import deepcopy

from xaikd import distillation_policies, bases, models, interceptor
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
    "parameterization",
    [
        "basis-center-rotationv2",
    ],
)
@pytest.mark.parametrize(
    "layerwise_training",
    [
        False,
        True,
    ],
)
def test_our_policies_callable(
    parameterization, basis_name, teacher_dims, student_dims, layerwise_training
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

        policy = distillation_policies.get_policy(
            parameterization, layerwise_training=layerwise_training, **kwargs
        )

        assert isinstance(
            policy,
            (
                distillation_policies.ours.OrthogonalBasisCenterRotationV2Policy,
                distillation_policies.ours.OrthogonalBasisCenterOrthoPolicy,
            ),
        )

        if layerwise_training:
            np.testing.assert_allclose(policy.scaling_factor, 1)
        else:
            np.testing.assert_allclose(
                policy.scaling_factor,
                basis.get_scale_factors_for_k(k=student_dims).sum(),
            )

        # check that this is callable
        policy(
            torch.randn(5, teacher_dims, 5, 5),
            torch.randn(5, student_dims, 5, 5),
        )

        if isinstance(
            policy,
            distillation_policies.ours.OrthogonalBasisCenterRotationV2Policy,
        ):
            expected_num_learnable_params = student_dims * student_dims
        elif isinstance(
            policy,
            distillation_policies.ours.OrthogonalBasisCenterOrthoPolicy,
        ):
            expected_num_learnable_params = 0
        else:
            raise

        _, actual_num_learnable_params = putils.count_params_in_list_params(
            policy.parameters()
        )

        np.testing.assert_allclose(
            actual_num_learnable_params, expected_num_learnable_params
        )

        assert True


def test_basis_identity_callable():
    teacher_dims = 10
    student_dims = 5
    basis_name = "identity"
    parameterization = "basis-center-ortho"
    layerwise_training = False

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

        policy = distillation_policies.get_policy(
            parameterization, layerwise_training=layerwise_training, **kwargs  # type: ignore
        )

        if layerwise_training:
            np.testing.assert_allclose(policy.scaling_factor, 1)  # type: ignore
        else:
            np.testing.assert_allclose(
                policy.scaling_factor,  # type: ignore
                basis.get_scale_factors_for_k(k=teacher_dims).sum(),
            )

        # check that this is callable
        policy(
            torch.randn(5, teacher_dims, 5, 5),
            torch.randn(5, student_dims, 5, 5),
        )

        expected_num_learnable_params = teacher_dims * student_dims

        _, actual_num_learnable_params = putils.count_params_in_list_params(
            policy.parameters()
        )

        np.testing.assert_allclose(
            actual_num_learnable_params, expected_num_learnable_params
        )

        assert True


@pytest.mark.skip(reason="obsolete")
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


@pytest.mark.parametrize(
    "policy_name,lambda_layer,expected_layer_policy,expected_collection",
    [
        (
            "student-only",
            00,
            "nothing",
            distillation_policies.LambdaCollection(
                lambda_task=1, lambda_kd=0, lambda_layer=0
            ),
        ),
        (
            "kd-only",
            100,
            "nothing",
            distillation_policies.LambdaCollection(
                lambda_task=0, lambda_kd=1, lambda_layer=0
            ),
        ),
        (
            "vid",
            10,
            "vid",
            distillation_policies.LambdaCollection(
                lambda_task=0, lambda_kd=1, lambda_layer=10
            ),
        ),
        (
            "attention-transfer",
            0.1,
            "attention-transfer",
            distillation_policies.LambdaCollection(
                lambda_task=0, lambda_kd=1, lambda_layer=0.1
            ),
        ),
        (
            "basis-bn-sum-normalized:prcaposdef-entropy0.95",
            0.1,
            "basis-bn-sum-normalized:prcaposdef-entropy0.95",
            distillation_policies.LambdaCollection(
                lambda_task=0, lambda_kd=1, lambda_layer=0.1
            ),
        ),
    ],
)
@pytest.mark.parametrize("layerwise_training", [True, False])
def test_resolve_lambdas_and_layer_policy(
    policy_name,
    lambda_layer,
    expected_layer_policy,
    expected_collection,
    layerwise_training,
):
    teacher = "cifar100-resnet18-v1"
    default_lambda_layer_config = None

    (
        actual_collection,
        actual_layer_policy,
    ) = distillation_policies.resolve_lambdas_and_layer_policy(
        teacher=teacher,
        policy_name=policy_name,
        lambda_layer=lambda_layer,
        default_lambda_layer_config=default_lambda_layer_config,
        layerwise_training=layerwise_training,
    )

    expected_collection = deepcopy(expected_collection)

    if layerwise_training and not policy_name in ["student-only", "kd-only"]:
        expected_collection.lambda_layer = 1

    np.testing.assert_equal(actual_layer_policy, expected_layer_policy)
    np.testing.assert_equal(actual_collection, expected_collection)


@pytest.mark.slow
@pytest.mark.parametrize(
    "teacher,student,layer_str",
    [
        (
            "imagenet-resnet101-tv",
            "student-resnet18-d16-16-8-8",
            "layer1:layer1,layer2:layer2,layer3:layer3,layer4:layer4",
        ),
        (
            "imagenet-resnet101-tv",
            "student-resnet18-d16-16-8-8",
            "layer1,layer2,layer3,layer4",
        ),
        (
            "imagenet-wideresnet101-tv",
            "student-resnet18-d16-16-8-8",
            "layer1,layer2,layer3,layer4",
        ),
        (
            "imagenet-resnet101-tv",
            "student-mobilenetv4-small",
            "layer1:blocks.1.1,layer2:blocks.2.3,layer3:blocks.3.1,layer4:blocks.3.5",
        ),
        (
            "imagenet-vitb-tv",
            "student-efficientformerv2_s0",
            "encoder.layers.2:stages.0,encoder.layers.5:stages.1,encoder.layers.8:stages.2,encoder.layers.11:stages.3",
        ),
    ],
)
def test_aligning_teacher_and_student_features_possible(
    teacher: str, student: str, layer_str
):
    arr_teacher_layers, arr_student_layers = distillation_policies.parse_layer_string(
        layer_str
    )

    device = putils.get_device()
    trng = torch.Generator()
    trng.manual_seed(1)

    x = torch.randn(5, 3, 224, 224, generator=trng).to(device)

    teacher_model = models.get_trained_model(teacher).to(device)

    _, arr_feat_teacher = interceptor.forward_and_intercept_intermediate_layers(
        model=teacher_model, inp=x, layers=arr_teacher_layers, detach_output=False
    )

    student_model = models.get_untrained_model(student, num_classes=10).to(device)

    _, arr_feat_student = interceptor.forward_and_intercept_intermediate_layers(
        model=student_model, inp=x, layers=arr_student_layers, detach_output=False
    )

    for feat_teacher, feat_student in zip(arr_feat_teacher, arr_feat_student):
        _, teacher_dims, _, _ = feat_teacher.shape
        _, student_dims, _, _ = feat_student.shape

        policy = distillation_policies.get_policy(
            "vid", teacher_dims=teacher_dims, student_dims=student_dims, device=device
        )

        loss = policy(feat_teacher, feat_student)

        assert loss is not None and torch.isfinite(loss)
