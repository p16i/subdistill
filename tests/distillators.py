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
    basis_name = "random"
    basis_mode = "centered"
    model_name = "cifar100-resnet18-p1"

    teacher_model = models.get_model(model_name)

    layers = ["layer3", "layer4"]
    student_layer_mapping = {"layer3": 128, "layer4": 256}

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

    for layer in layers:
        layer_policies.append(
            (
                layer,
                criteria.LearnableLinL2Loss(
                    teacher_dims=constants.ARCH_LAYER_DIMENSIONS["resnet18"][layer],
                    student_dims=student_layer_mapping[layer],
                    device=device,
                ),
            )
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
            student=models._pat_resnet(num_classes=dataset.num_classes),
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
            get_batchnorm_statistics_from_model(models.get_model(model_name)),
        ):
            np.testing.assert_allclose(
                after_bn_stat,
                before_bn_stat,
                err_msg="BatchNorm stat stay the same!",
            )


# todo: add test for _get_paramaters
