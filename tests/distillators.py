import os
import pytest

import tempfile

import numpy as np

from pathlib import Path

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


def test_checking_no_batchnorm_get_updated_during_distillation():
    layer = "layer3"
    model_name = "cifar100-resnet18-p1"
    teacher_model = models.get_model(model_name)
    dataset = datasets.construct("cifar100-people")
    device = "cpu"

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

    compression_ratio = 1.0
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

        before_bn1_mean = student.bn1.running_mean.clone().numpy()
        before_bn2_mean = student.layer4[0].bn1.running_mean.clone().numpy()

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
        )

        log_dir = tmpdirname / "distillation" / "log"

        os.makedirs(log_dir, exist_ok=True)

        for param in layer_approximator.parameters():
            param.requires_grad = True

        print(
            f"[before init] we have {utils.count_params_in_model(layer_approximator)} parameters (id={id(layer_approximator)})"
        )

        results = distillator.distill(
            student=student,
            approx_mod=layer_approximator,
            distill_info=distill_info,
            epochs=1,
            basis=basis,
            device=device,
            lr=0.001,
            log_dir=log_dir,
            lambda_mse=1.0,
            lambda_xent=1.0,
        )

        after_bn1_mean = student.bn1.running_mean.clone().numpy()
        after_bn2_mean = student.layer4[0].bn1.running_mean.clone().numpy()

        for before, after in [
            (before_bn1_mean, after_bn1_mean),
            (before_bn2_mean, after_bn2_mean),
        ]:
            np.testing.assert_allclose(
                after, before, err_msg="Batchnorm stats changes!"
            )
