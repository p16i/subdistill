import click
import os
import pandas as pd

import pytorch_lightning as pl
import numpy as np

from datetime import datetime

from pathlib import Path
from copy import deepcopy

import torch
from tqdm import tqdm
import wandb

from xaikd import (
    datasets,
    utils,
    distillators,
    models,
    attributors,
    bases,
    approximators,
    distillation_info,
)

from xaikd import criteria

from xaikd.approximators import ApproximatorMode

from xaikd.distillation_info import ExperimentConfiguration

from pytorch_lightning.loggers import WandbLogger


WANDB_PROJECT = os.getenv("WANDB_PROJECT", "xaikd-distillation-layerwise")
LAYERS = ["layer1", "layer2", "layer3", "layer4"]


@click.command()
@click.option("--dataset", default="cifar100-people", type=str, required=True)
@click.option("--teacher-model", default="cifar100-resnet18-wb15", required=True)
@click.option("--layer", default="layer3", type=str, required=True)
@click.option(
    "--basis-names",
    type=str,
    default="pca,prca-recon,prca-sortabs",
    required=True,
)
@click.option("--basis-mode", type=str, default="centered", required=True)
# @click.option("--compression-ratio", type=float, default=4.0, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--training-size", type=float, default=0.1, required=True)
@click.option("--epochs", type=int, default=100, required=True)
@click.option("--seed", type=int, default=1)
@click.option("--lr", type=float, default=0.0005, required=True)
@click.option("--lambda-layer", type=float, default=1.0)
@click.option("--lambda-kd", type=float, default=1.0)
@click.option("--lambda-task", type=float, default=0.0)
@click.option("--skip-if-exist", type=bool, default=False, is_flag=True)
@click.option("--skip-baselines", type=bool, default=False, is_flag=True)
def main(
    teacher_model,
    dataset,
    basis_names,
    output_dir,
    seed,
    epochs,
    lr,
    training_size,
    layer,
    basis_mode,
    lambda_task,
    lambda_kd,
    lambda_layer,
    skip_baselines,
    skip_if_exist,
):
    arguments = locals()

    pl.seed_everything(seed)

    start_time = datetime.now()

    teacher_model = models.get_model(teacher_model)

    model_name = getattr(teacher_model, "__name")

    basis_names = basis_names.split(",")

    output_dir = (
        Path(output_dir) / f"{dataset}-tz{training_size}-seed{seed}" / model_name
    )

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    dataset = datasets.construct(dataset)

    teacher_model.to(device)

    logit_mod = attributors.OneClassEvidence(num_classes=dataset.num_classes)

    ds_train = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=training_size, seed=seed
    )

    train_loader = datasets.build_dataloader(ds_train, shuffle=True)
    val_loader = datasets.build_dataloader(
        dataset.create_subset(train_split=False),
        shuffle=False,
    )

    ds_train_with_aug = deepcopy(ds_train)
    ds_train_with_aug.dataset.transform = dataset.input_training_transformation

    train_loader_with_aug = datasets.build_dataloader(
        ds_train_with_aug, shuffle=True, batch_size=int(np.ceil(64 * training_size))
    )

    utils.modify_last_layer_for_subclasses(
        teacher_model.fc, selected_classes=dataset.selected_classes
    )

    STUDENT_MODEL_NAME = "resnet18-2"

    ARR_LAYERS = ["layer1", "layer2", "layer3", "layer4"][2:]
    ARR_DIMS = [32, 64, 128, 256][2:]

    # todo: to be remove
    # model_student = models._pat_resnet(num_classes=dataset.num_classes)
    # dummy_input = torch.randn(64, 3, 32, 32)
    # model_student(dummy_input)

    # print(teacher_model)
    # print(model_student)

    # prepare teacher to have the logit equal to num classes

    for layer in ARR_LAYERS:
        layer_output_dir = output_dir / f"layer-{layer}"
        os.makedirs(layer_output_dir, exist_ok=True)
        arr_act, arr_ctx = attributors.extract_activation_context(
            model=teacher_model,
            layer=layer,
            data_loader=train_loader,
            dataset=dataset,
            logit_modifier=logit_mod,
            device=device,
            rng=np.random.default_rng(seed=seed),
        )

        mean = np.mean(arr_act, axis=0)
        np.save(layer_output_dir / "act_mean", mean)

        for basis_name in basis_names:
            click.echo(f"[layer={layer}] fitting basis={basis_name}--{basis_mode}")
            basis = bases.get_basis(f"{basis_name}--{basis_mode}", seed=seed)
            basis.fit(arr_act, arr_ctx, mean=mean, device=device)
            basis.save(layer_output_dir)

    for basis_name in tqdm(basis_names, desc="Distillation"):
        layer_policies = []
        for dim, layer in zip(ARR_DIMS, ARR_LAYERS):
            basis = bases.get_basis(f"{basis_name}--{basis_mode}", seed=seed)
            layer_output_dir = output_dir / f"layer-{layer}"
            basis.load(layer_output_dir)

            layer_policies.append(
                (
                    layer,
                    criteria.BasisL2Loss(basis, dim, device),
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

        log_dir = output_dir / "distilled-models" / STUDENT_MODEL_NAME
        logger = WandbLogger(
            project=WANDB_PROJECT,
            group=arguments["output_dir"],
            job_type="distillation",
            name=f"{STUDENT_MODEL_NAME}-{basis_name}-seed{seed}",
            config={
                **arguments,
                "basis_name": basis_name,
                "output_dir": output_dir,
            },
        )

        student, results = distillator.distill(
            student=models._pat_resnet(num_classes=dataset.num_classes),
            layer_policies=layer_policies,
            epochs=epochs,
            lambda_task=lambda_task,
            lambda_kd=lambda_kd,
            lambda_layer=lambda_layer,
            device=device,
            lr=lr,
            log_dir=log_dir,
            logger=logger,
        )

        last_epoch_val_acc = results["arr_metrics"]["val"][-1]

        print(f"Result: Student with  `{basis_name}` acc={last_epoch_val_acc:.4f}")

        for k, v in results.items():
            logger.experiment.summary[k] = v

        wandb.finish()

        """
        Distillator(layer_policies=[
            ('layer3', criteria(teacher_feat, student_feat)),
        ])

        Distllator.distill(
            epochs,
            logger=...
            lambda_task,
            lambda_kd,
            lambda_layer
        )

        tests:
        - teacher remain the same
        - distill same seed twice give same results.
        """
        # g
        # lambda_task, lambda_

    # for each layer, we learn basis

    # for each basis
    # we do distillation
    # get student model

    click.echo(f"Check Results at: {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")

    return

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=teacher_model,
        layer=layer,
        data_loader=train_loader,
        dataset=dataset,
        logit_modifier=logit_mod,
        device=device,
        rng=np.random.default_rng(seed=seed),
    )
    mean = np.mean(arr_act, axis=0)
    # todo: add overwriting flag; if exist, not overwrite and assert!
    np.save(output_dir / "act_mean", mean)

    distill_info = distillation_info.get_distill_infor(
        arch=model_name, layer=layer, compression_ratio=compression_ratio
    )

    ref_acc = None

    arr_experiment_confs = []

    for basis_name in basis_names.split(","):
        arr_experiment_confs.append(
            ExperimentConfiguration(
                basis_name=f"{basis_name}--{basis_mode}",
                compression_ratio=compression_ratio,
                approximator_mode=ApproximatorMode.HOMOGENOUS_LOWRANK,
            ),
        )

    if not skip_baselines:
        arr_experiment_confs.extend(
            [
                ExperimentConfiguration(
                    basis_name=f"identity--{basis_mode}",
                    compression_ratio=1.0,
                    approximator_mode=ApproximatorMode.HOMOGENOUS,
                ),
                ExperimentConfiguration(
                    basis_name=f"identity--{basis_mode}",
                    compression_ratio=compression_ratio,
                    approximator_mode=ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER,
                ),
            ]
        )

    for conf in tqdm(arr_experiment_confs):
        conf: ExperimentConfiguration
        approximator = approximators.construct_approximator_for(
            teacher_model,
            layer=layer,
            compression_ratio=conf.compression_ratio,
            mode=conf.approximator_mode,
            seed=seed,
        )

        distillator = distillators.Layerwise(
            teacher=models.get_model(model_name),
            dataset=dataset,
            train_dataloader=train_loader_with_aug,
            val_dataloader=val_loader,
            device=device,
            weight_decay=weight_decay,
        )

        if ref_acc is None:
            ref_acc = distillator.ref_acc
        else:
            assert (
                distillator.ref_acc == ref_acc
            ), "Reference models have different accuracy!"

        basis_name = conf.basis_name

        approximator_mode = approximators.normalize_mode_name(conf.approximator_mode)

        basis_distillation_output_dir = (
            output_dir
            / "distillation"
            / f"{approximator_mode}-comp{conf.compression_ratio}-wd{weight_decay}-ldmse{lambda_mse}-ldxent{lambda_xent}"
            / basis_name
        )

        if skip_if_exist and os.path.exists(basis_distillation_output_dir):
            click.echo(
                f"Directory `{basis_distillation_output_dir}` already exists! Skipping the task"
            )
            continue

        os.makedirs(basis_distillation_output_dir, exist_ok=True)

        basis = bases.get_basis(basis_name, seed=seed)
        #  todo: only fit if necessary
        basis.fit(arr_act, arr_ctx, mean=mean, device=device)
        basis.save(output_dir)
        basis.load(output_dir)

        student = models.get_model(model_name)

        logger = WandbLogger(
            project=WANDB_PROJECT,
            group=arguments["output_dir"],
            job_type="distillation",
            name=f"{conf.basis_name}-compr{conf.compression_ratio}-seed{seed}",
            config={
                **arguments,
                "approximator_mode": approximator_mode,
                "compression_ratio": conf.compression_ratio,
                "basis_name": basis_name,
                "approximator": f"{conf}",
                "output_dir": output_dir,
            },
        )

        student, results = distillator.distill(
            student=student,
            approximator=approximator,
            distill_info=distill_info,
            epochs=epochs,
            basis=basis,
            device=device,
            lr=lr,
            logger=logger,
            log_dir=basis_distillation_output_dir / "log",
            lambda_mse=lambda_mse,
            lambda_xent=lambda_xent,
        )

        last_epoch_val_acc = results["arr_metrics"]["val"][-1]

        print(
            f"Result: Student with `{approximator_mode}` and `{basis}` acc={last_epoch_val_acc:.4f}"
        )

        # dumps results to wandb
        for k, v in results.items():
            logger.experiment.summary[k] = v

        utils.dump_json(basis_distillation_output_dir / "results.json", results)


if __name__ == "__main__":
    main()
