import typing
import os
import pytest

import torch

from collections import OrderedDict

from torch import nn

import tempfile

import numpy as np

from pathlib import Path

from copy import deepcopy

from pytorch_lightning.loggers.wandb import WandbLogger

from xaikd import (
    distillation_policies,
    distillators,
    models,
    constants,
    utils,
)

from xaikd import datasets, metrics
from xaikd.distillation_policies.interface import LayerPolicyCollection


def get_batchnorm_statistics_from_model(model: nn.Module) -> typing.List[torch.Tensor]:
    stats = []
    for bn in utils.query_module_children_with_type(model, nn.BatchNorm2d):

        assert not bn.running_mean is None
        stats.append(bn.running_mean.clone().cpu().numpy())

        assert not bn.running_var is None
        stats.append(bn.running_var.clone().cpu().numpy())

    return stats


@pytest.mark.gpu()
@pytest.mark.slow()
@pytest.mark.parametrize(
    "teacher_model_name,layers",
    [
        (
            "cifar100-resnet18-v1",
            "layer1,layer2,layer3,layer4",
        ),
        (
            "cifar100-resnet50-v1",
            "layer1,layer2,layer3,layer4",
        ),
    ],
)
def test_distillation_runnable_and_correct(
    teacher_model_name,
    layers,
):

    last_layer_policy = distillation_policies.get_last_layer_policy("last-layer:binkd")

    epochs = 1
    teacher_layers, student_layers = distillation_policies.parse_layer_string(layers)

    dataset = datasets.construct("cifar100-people-vs-others")

    teacher_model = nn.Sequential(
        OrderedDict(
            [
                ("base", models.get_trained_model(teacher_model_name)),
                (
                    "logodd",
                    models.layers.LayerLogOddSelectedClasses(
                        selected_classes=dataset.selected_classes
                    ),
                ),
            ]
        )
    )
    teacher_model.eval()

    teacher_layers = list(map(lambda l: f"base.{l}", teacher_layers))

    device = utils.get_device()

    ds_training = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=0.05, seed=1
    )
    ds_val = datasets.subsample_dataset(
        dataset.create_subset(train_split=False), ratio=0.05, seed=1
    )

    train_loader = datasets.build_dataloader(
        ds_training,
        shuffle=True,
        persistent_workers=False,
        pin_memory=False,
        num_workers=1,
    )
    val_loader = datasets.build_dataloader(
        ds_val, shuffle=False, persistent_workers=False, pin_memory=False, num_workers=1
    )

    teacher_dims_mapping = utils.get_dimensions_at_layers(
        teacher_model, train_loader, teacher_layers
    )
    student_dims_mapping = utils.get_dimensions_at_layers(
        models.get_untrained_model(
            constants.STUDENT_MODEL_FOR_TESTING, num_classes=dataset.num_classes
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
        teacher=teacher_model,
        dataset=dataset,
        dataloader_train=train_loader,
        dataloader_val=val_loader,
        dataloader_test=val_loader,
        device=device,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        logger = WandbLogger(save_dir=tmpdirname, project="unittest", offline=True)
        student = distillator.distill(
            student=models.get_untrained_model(
                constants.STUDENT_MODEL_FOR_TESTING, num_classes=dataset.num_classes
            ),
            last_layer_policy=last_layer_policy,
            layer_policies=layer_policies,
            epochs=epochs,
            lambda_task=1.0,
            lambda_kd=1.0,
            lambda_layer=1.0,
            device=device,
            lr=1e-4,
            weight_decay=0,
            logger=logger,
            seed=1,
            upload_best_checkpoint=False,
        )

    # post-training assertions
    with torch.no_grad():
        metric = metrics.MetricAUROCBinaryCrossEntropy()
        # sanity check `student``
        actual_auroc, _ = metric(
            student,
            dataloader=val_loader,
            device=device,
        )

        expected_auroc = logger.experiment.summary["student_best_val_auroc"]
        np.testing.assert_allclose(actual_auroc, expected_auroc)

        # sanity check `teacher`
        expected_teacher_metric = metric(
            teacher_model_before.to(device),
            dataloader=val_loader,
            device=device,
        )
        np.testing.assert_allclose(
            (distillator.ref_auroc, distillator.ref_xent), expected_teacher_metric
        )

        # ucheck teacher parameters not get updated!
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
def test_get_parameters(layers):
    dataset = datasets.construct("cifar100-people")

    device = utils.get_device()

    layer = "layer3"

    last_layer_policy = distillation_policies.get_last_layer_policy("last-layer:binkd")

    teacher_model = models.get_trained_model("cifar100-resnet18-v1")
    student = models.get_untrained_model(
        constants.STUDENT_MODEL_FOR_TESTING, num_classes=dataset.num_classes
    )

    train_loader = datasets.build_dataloader(
        dataset.create_subset(train_split=True),
        shuffle=False,
        persistent_workers=False,
        pin_memory=False,
        num_workers=1,
    )

    teacher_dims_mapping = utils.get_dimensions_at_layers(
        teacher_model, train_loader, layers
    )
    student_dims_mapping = utils.get_dimensions_at_layers(
        models.get_untrained_model(
            constants.STUDENT_MODEL_FOR_TESTING, num_classes=dataset.num_classes
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

    layer_policy_collection = LayerPolicyCollection(
        teacher_layers=layers, student_layers=layers, policies=adapters
    )

    model_training_wrapper = distillators.LayerwiseKDModelWrapper(
        teacher=teacher_model,
        student=student,
        last_layer_policy=last_layer_policy,
        layerwise_policies=layer_policy_collection,
        lambda_kd=1,
        lambda_layer=1,
        lambda_task=1,
        weight_decay=0,
        lr=1e-5,
    )

    actual_num_params = utils.count_params_in_list_params(
        model_training_wrapper._get_parameters()
    )

    expected_params = list(student.parameters())
    for adapter in adapters:
        expected_params.extend(list(adapter.parameters()))

    expected_num_params = utils.count_params_in_list_params(expected_params)

    assert actual_num_params == expected_num_params
