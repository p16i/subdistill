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
    criteria,
    distillators,
    models,
    datasets,
    constants,
    utils,
)

from xaikd.utils import metrics


def get_batchnorm_statistics_from_model(model: nn.Module) -> typing.List[torch.Tensor]:
    stats = []
    for bn in utils.query_module_children_with_type(model, nn.BatchNorm2d):
        stats.append(bn.running_mean.clone().cpu().numpy())

        stats.append(bn.running_var.clone().cpu().numpy())

    return stats


@pytest.mark.gpu()
@pytest.mark.slow()
def test_distillation_not_alter_teacher():
    teacher_model_name = "cifar100-resnet18-p1"
    student_model_name = "resnet18compr2"

    teacher_model = models.get_trained_model(teacher_model_name)

    layers = ["layer3", "layer4"]

    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        "cifar100-people"
    )

    utils.modify_last_layer_for_subclasses(teacher_model.fc, dataset.selected_classes)

    device = utils.get_device()

    ds_training = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=0.05, seed=1
    )
    ds_val = datasets.subsample_dataset(
        dataset.create_subset(train_split=False), ratio=0.05, seed=1
    )

    train_loader = datasets.build_dataloader(ds_training, shuffle=True)
    val_loader = datasets.build_dataloader(ds_val, shuffle=False)

    np.random.seed(1)

    layer_policies = []

    arr_adapters = []
    for layer in layers:
        arr_adapters.append(
            criteria.LearnableLinL2Loss(
                teacher_dims=constants.ARCH_LAYER_DIMENSIONS["resnet18"][layer],
                student_dims=constants.ARCH_LAYER_DIMENSIONS[student_model_name][layer],
                device=device,
            )
        )

    layer_policies = distillators.LayerPolicyCollection(
        layers=layers, policies=arr_adapters
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
        )

    # post-training assertions
    with torch.no_grad():
        actual_acc, _ = metrics.accuracy(
            student,
            dataloader=val_loader,
            num_classes=dataset.num_classes,
            device=device,
        )
        expected_acc = results["arr_metrics"]["val"][-1]
        np.testing.assert_allclose(actual_acc, expected_acc)

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

        # check batchnorm stats before and after
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


# todo: add test for _get_paramaters


@pytest.mark.parametrize("layers", [["layer3"], ["layer3", "layer4"]])
def test_get_parameters(layers):
    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        "cifar100-people"
    )

    device = utils.get_device()

    layer = "layer3"

    teacher_model = models.get_trained_model("cifar100-resnet18-p1")
    student = models.get_untrained_model(
        "resnet18compr2", num_classes=dataset.num_classes
    )

    adapters = []
    for layer in layers:
        dim = constants.ARCH_LAYER_DIMENSIONS["resnet18compr2"][layer]
        adapters.append(
            criteria.LearnableLinL2Loss(
                teacher_dims=constants.ARCH_LAYER_DIMENSIONS["resnet18"][layer],
                student_dims=dim,
                device=device,
            )
        )

    layer_policy_colleciton = distillators.LayerPolicyCollection(
        layers=layers, policies=adapters
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
    )

    actual_num_params = utils.count_params_in_list_params(
        model_training_wrapper._get_parameters()
    )

    expected_params = list(student.parameters())
    for adapter in adapters:
        expected_params.extend(list(adapter.parameters()))

    expected_num_params = utils.count_params_in_list_params(expected_params)

    assert actual_num_params == expected_num_params
