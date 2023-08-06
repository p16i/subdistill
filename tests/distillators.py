import os
import pytest

import torch

from torch import nn

import tempfile

import numpy as np

from pathlib import Path

from copy import deepcopy

from xaikd import (
    approximators,
    bases,
    distillators,
    models,
    datasets,
    constants,
    bases,
    distillation_info,
    utils,
)

from xaikd.utils import metrics


@pytest.mark.gpu()
@pytest.mark.slow()
@pytest.mark.parametrize("layer", ["layer3", "layer4"])
@pytest.mark.parametrize("compression_ratio", [1.0, 2.0])
def test_distillation_not_alter_batchnorm_and_other_params(layer, compression_ratio):
    model_name = "cifar100-resnet18-p1"
    teacher_model = models.get_model(model_name)
    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        "cifar100-people"
    )
    device = utils.get_device()

    layer_dim = constants.ARCH_LAYER_DIMENSIONS["resnet18"][layer]

    ds_training = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=0.05, seed=1
    )
    ds_val = datasets.subsample_dataset(
        dataset.create_subset(train_split=False), ratio=0.05, seed=1
    )

    train_loader = datasets.build_dataloader(ds_training, shuffle=True)
    val_loader = datasets.build_dataloader(ds_val, shuffle=False)

    basis = bases.get_basis("identity--centered")

    np.random.seed(1)

    approximator_mode = approximators.ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER
    distill_info = distillation_info.get_distill_infor(
        arch=model_name, layer=layer, compression_ratio=compression_ratio
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        arr_act = np.random.randn(32, layer_dim) + 1
        mean = np.mean(arr_act, axis=0)
        np.save(tmpdirname / "act_mean", mean)
        basis.fit(arr_act, None, mean=mean, device=device)
        basis.save(tmpdirname)
        basis.load(tmpdirname)

        student = models.get_model(model_name)

        expected_acc = metrics.accuracy_with_subclasses(
            student,
            val_loader,
            dataset.selected_classes,
            dataset.transform_target,
            device=device,
        )

        (
            before_feature_extractor,
            before_approx,
            before_classification_head,
        ) = models.resnet.split_resnet_18_at(student, layer=layer)

        np.testing.assert_allclose(
            metrics.accuracy_with_subclasses(
                nn.Sequential(
                    before_feature_extractor, before_approx, before_classification_head
                ),
                val_loader,
                dataset.selected_classes,
                dataset.transform_target,
                device=device,
            ),
            expected_acc,
        )

        before_modules = list(
            map(
                lambda m: deepcopy(m),
                [teacher_model, before_feature_extractor, before_classification_head],
            )
        )

        before_batch_norm_stats = list(
            map(
                lambda bn: bn.running_mean.clone().cpu().numpy(),
                utils.query_module_children_with_type(
                    nn.Sequential(*before_modules), nn.BatchNorm2d
                ),
            )
        )

        distillator = distillators.Layerwise(
            teacher=teacher_model,
            dataset=dataset,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=device,
            weight_decay=0.0,
        )

        layer_approximator = approximators.construct_approximator_for(
            teacher_model,
            layer=layer,
            compression_ratio=compression_ratio,
            mode=approximator_mode,
            seed=1,
        )

        log_dir = tmpdirname / "distillation" / "log"

        results = distillator.distill(
            student=student,
            approximator=layer_approximator,
            distill_info=distill_info,
            epochs=1,
            basis=basis,
            device=device,
            lr=0.001,
            log_dir=log_dir,
            lambda_mse=1.0,
            lambda_xent=1.0,
        )

        (
            after_feature_extractor,
            after_approx,
            after_classification_head,
        ) = models.resnet.split_resnet_18_at(student, layer=layer)

        after_modules = [
            teacher_model,
            after_feature_extractor,
            after_classification_head,
        ]

        after_batch_norm_stats = list(
            map(
                lambda bn: bn.running_mean.clone().cpu().numpy(),
                utils.query_module_children_with_type(
                    nn.Sequential(*after_modules), nn.BatchNorm2d
                ),
            )
        )

        np.testing.assert_allclose(
            metrics.accuracy_with_subclasses(
                nn.Sequential(
                    after_feature_extractor, before_approx, after_classification_head
                ),
                val_loader,
                dataset.selected_classes,
                dataset.transform_target,
                device=device,
            ),
            expected_acc,
        )

        with torch.no_grad():
            for before_bn_stat, after_bn_stat in zip(
                before_batch_norm_stats, after_batch_norm_stats
            ):
                np.testing.assert_allclose(
                    before_bn_stat,
                    after_bn_stat,
                    err_msg="BatchNorm stat stay the same!",
                )

            for before, after in zip(
                before_modules,
                after_modules,
            ):
                before_total_params, _ = utils.count_params_in_model(before)
                after_total_params, _ = utils.count_params_in_model(after)

                assert before_total_params == after_total_params

                for before_params, after_params in zip(
                    before.parameters(), after.parameters()
                ):
                    np.testing.assert_allclose(
                        before_params.cpu(),
                        after_params.cpu(),
                        err_msg="All parameters stay the same",
                    )

            with pytest.raises(AssertionError):
                for before_approx_param, after_approx_param in zip(
                    before_approx.parameters(), after_approx.parameters()
                ):
                    np.testing.assert_allclose(
                        before_approx_param.cpu(), after_approx_param.cpu()
                    )
