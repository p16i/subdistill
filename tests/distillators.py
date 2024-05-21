import typing
import os
import pytest

import torch

from torch import nn

import tempfile

import numpy as np

from pathlib import Path

from copy import deepcopy

from pytorch_lightning.loggers import TensorBoardLogger

from xaikd import (
    distillation_policies,
    distillators,
    models,
    datasets,
    constants,
    utils,
)

from xaikd.utils import metrics
from xaikd.distillation_policies import LayerPolicyCollection


def get_batchnorm_statistics_from_model(model: nn.Module) -> typing.List[torch.Tensor]:
    stats = []
    for bn in utils.query_module_children_with_type(model, nn.BatchNorm2d):
        stats.append(bn.running_mean.clone().cpu().numpy())

        stats.append(bn.running_var.clone().cpu().numpy())

    return stats


@pytest.mark.gpu()
@pytest.mark.slow()
@pytest.mark.parametrize(
    "teacher_model_name,student_model_name,layers",
    [
        (
            "cifar100-resnet18-v1",
            "resnet18xscifarcompr2",
            "layer1,layer2,layer3,layer4",
        ),
        (
            "cifar100-resnet50-v1",
            "resnet18xscifarcompr1",
            "layer1,layer2,layer3,layer4",
        ),
        (
            "cifar100-vgg11-v1",
            "vgg8xs",
            "features.10:features.8,features.15:features.11,features.20:features.14",
        ),
        (
            "cifar100-resnet18-v1",
            "vgg8xs",
            "layer3:features.8",
        ),
    ],
)
@pytest.mark.parametrize("detach_output", [False, True])
def test_distillation_runnable_and_correct(
    teacher_model_name, student_model_name, layers, detach_output
):
    teacher_layers, student_layers = distillation_policies.parse_layer_string(layers)

    teacher_model = models.get_trained_model(teacher_model_name)

    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        "cifar100-people"
    )

    utils.modify_last_layer_for_subclasses(teacher_model, dataset.selected_classes)

    device = utils.get_device()

    ds_training = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=0.05, seed=1
    )
    ds_val = datasets.subsample_dataset(
        dataset.create_subset(train_split=False), ratio=0.05, seed=1
    )

    train_loader = datasets.build_dataloader(ds_training, shuffle=True)
    val_loader = datasets.build_dataloader(ds_val, shuffle=False)

    teacher_dims_mapping = utils.get_dimensions_at_layers(
        teacher_model, train_loader, teacher_layers
    )
    student_dims_mapping = utils.get_dimensions_at_layers(
        models.get_untrained_model(
            student_model_name, num_classes=dataset.num_classes
        ).eval(),
        train_loader,
        student_layers,
    )

    np.random.seed(1)

    layer_policies = []

    arr_policies = []
    for teacher_layer, student_layer in zip(teacher_layers, student_layers):
        arr_policies.append(
            distillation_policies.FitNet(
                teacher_dims=teacher_dims_mapping[teacher_layer],
                student_dims=student_dims_mapping[student_layer],
                device=device,
            )
        )

    layer_policies = LayerPolicyCollection(
        teacher_layers=teacher_layers,
        student_layers=student_layers,
        policies=arr_policies,
    )

    teacher_model_before = deepcopy(teacher_model)
    before_batch_norm_stats = get_batchnorm_statistics_from_model(teacher_model_before)

    distillator = distillators.Layerwise(
        teacher=teacher_model.train(),
        dataset=dataset,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        weight_decay=0.0,
        detach_layer_output_in_forward_hook=detach_output,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        student, results = distillator.distill(
            student=models.get_untrained_model(
                student_model_name, num_classes=dataset.num_classes
            ),
            layer_policies=layer_policies,
            epochs=1,
            lambda_task=1.0,
            lambda_kd=1.0,
            lambda_layer=1.0,
            device=device,
            lr=1e-4,
            log_dir=Path(tmpdirname),
            logger=TensorBoardLogger(tmpdirname),
            seed=1,
            enable_checkpointing=False,
        )

    # post-training assertions
    with torch.no_grad():
        # sanity check `student``
        actual_acc, _ = metrics.accuracy(
            student,
            dataloader=val_loader,
            num_classes=dataset.num_classes,
            device=device,
        )
        expected_acc = results["arr_metrics"]["val_acc"][-1]
        np.testing.assert_allclose(actual_acc, expected_acc)

        # sanity check `teacher`
        expected_teacher_acc_xent = metrics.accuracy(
            teacher_model_before.to(device),
            dataloader=val_loader,
            num_classes=dataset.num_classes,
            device=device,
        )
        np.testing.assert_allclose(
            (distillator.ref_acc, distillator.ref_xent), expected_teacher_acc_xent
        )

        # check teacher parameters not get updated!
        for before_params, after_params in zip(
            teacher_model_before.parameters(), teacher_model.parameters()
        ):
            np.testing.assert_allclose(
                after_params.cpu(),
                before_params.cpu(),
                err_msg="All parameters stay the same",
            )

        after_batch_norm_stats = get_batchnorm_statistics_from_model(teacher_model)

        # check batchnorm stats of teacher before and after
        for before_bn_stat, after_bn_stat in zip(
            before_batch_norm_stats, after_batch_norm_stats
        ):
            np.testing.assert_allclose(
                after_bn_stat,
                before_bn_stat,
                err_msg="BatchNorm stat stay the same!",
            )

        # check batchnorm stats before and newly instatiated teacher
        for before_bn_stat, after_bn_stat in zip(
            before_batch_norm_stats,
            get_batchnorm_statistics_from_model(
                models.get_trained_model(teacher_model_name)
            ),
        ):
            np.testing.assert_allclose(
                after_bn_stat,
                before_bn_stat,
                err_msg="BatchNorm stat stay the same!",
            )


@pytest.mark.parametrize("layers", [["layer3"], ["layer3", "layer4"]])
@pytest.mark.parametrize("detach_output", [True, False])
def test_get_parameters(layers, detach_output):
    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        "cifar100-people"
    )

    device = utils.get_device()

    layer = "layer3"

    teacher_model = models.get_trained_model("cifar100-resnet18-v1")
    student = models.get_untrained_model(
        "resnet18xscifarcompr2", num_classes=dataset.num_classes
    )

    train_loader = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )

    teacher_dims_mapping = utils.get_dimensions_at_layers(
        teacher_model, train_loader, layers
    )
    student_dims_mapping = utils.get_dimensions_at_layers(
        models.get_untrained_model(
            "resnet18xscifarcompr2", num_classes=dataset.num_classes
        ).eval(),
        train_loader,
        layers,
    )

    adapters = []
    for layer in layers:
        adapters.append(
            distillation_policies.FitNet(
                teacher_dims=teacher_dims_mapping[layer],
                student_dims=student_dims_mapping[layer],
                device=device,
            )
        )

    layer_policy_colleciton = LayerPolicyCollection(
        teacher_layers=layers, student_layers=layers, policies=adapters
    )

    model_training_wrapper = distillators.LayerwiseKDModelWrapper(
        teacher=teacher_model,
        student=student,
        layerwise_policies=layer_policy_colleciton,
        lambda_kd=1,
        lambda_layer=1,
        lambda_task=1,
        lr=1e-5,
        num_classes=dataset.num_classes,
        detach_layer_output_in_forward_hook=detach_output,
    )

    actual_num_params = utils.count_params_in_list_params(
        model_training_wrapper._get_parameters()
    )

    expected_params = list(student.parameters())
    for adapter in adapters:
        expected_params.extend(list(adapter.parameters()))

    expected_num_params = utils.count_params_in_list_params(expected_params)

    assert actual_num_params == expected_num_params
