import click
import os
import pandas as pd

import pytorch_lightning as pl
import numpy as np

from datetime import datetime

from pathlib import Path
from copy import deepcopy

from torchvision import transforms
from tqdm import tqdm

from xaikd import (
    datasets,
    utils,
    distillators,
    models,
    attributors,
    bases,
    approximators,
    augmentations,
    distillation_info,
)

from xaikd.approximators import ApproximatorMode

from xaikd.distillation_info import ExperimentConfiguration


@click.command()
@click.option("--dataset", default="cifar100-people", type=str, required=True)
@click.option("--teacher-model", default="cifar100-resnet18-p1", required=True)
@click.option("--layer", default="layer3", type=str, required=True)
@click.option(
    "--basis-names",
    type=str,
    default="pca,prca-recon,prca-abs,pcaprca-abs,pcaprca-recon,random1",
    required=True,
)
@click.option("--basis-mode", type=str, default="centered", required=True)
@click.option("--compression-ratio", type=float, default=4.0, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--training-size", type=float, default=0.1, required=True)
@click.option("--epochs", type=int, default=100, required=True)
@click.option("--lr", type=float, default=0.0005, required=True)
@click.option("--seed", type=int, default=1)
@click.option("--weight-decay", type=float, default=0.0)
@click.option("--lambda-mse", type=float, default=1.0)
@click.option("--lambda-xent", type=float, default=1.0)
@click.option("--skip-if-exist", type=bool, default=False, is_flag=True)
def main(
    teacher_model,
    dataset,
    basis_names,
    output_dir,
    compression_ratio,
    seed,
    epochs,
    lr,
    training_size,
    layer,
    basis_mode,
    weight_decay,
    lambda_mse,
    lambda_xent,
    skip_if_exist,
):
    pl.seed_everything(seed)

    arguments = locals()
    start_time = datetime.now()

    teacher_model = models.get_model(teacher_model)
    model_name = getattr(teacher_model, "__name")

    lr = lr / (training_size)

    layer_slug = f"layer-{layer}"

    output_dir = (
        Path(output_dir)
        / f"{dataset}-tz{training_size}-seed{seed}"
        / model_name
        / layer_slug
    )

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    dataset = datasets.construct(dataset)

    teacher_model.to(device)

    logit_mod = attributors.OneClassEvidence(dataset=dataset)

    ds_train = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=training_size, seed=seed
    )

    train_loader = datasets.build_dataloader(ds_train, shuffle=True)
    val_loader = datasets.build_dataloader(
        dataset.create_subset(train_split=False),
        shuffle=False,
    )

    ds_train_with_aug = deepcopy(ds_train)
    ds_train_with_aug.dataset.transform = transforms.Compose(
        [
            *augmentations.get_augmentation_for(dataset=dataset),
            ds_train_with_aug.dataset.transform,
        ]
    )

    train_loader_with_aug = datasets.build_dataloader(ds_train_with_aug, shuffle=True)

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

    arr_experiment_confs = [
        # todo: make sure that we run this conf only once!
        ExperimentConfiguration(
            basis_name="identity--uncentered",
            compression_ratio=1.0,
            approximator_mode=ApproximatorMode.HOMOGENOUS,
        ),
        ExperimentConfiguration(
            basis_name="identity--uncentered",
            compression_ratio=compression_ratio,
            approximator_mode=ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER,
        ),
    ]

    for basis_name in basis_names.split(","):
        arr_experiment_confs.append(
            ExperimentConfiguration(
                basis_name=f"{basis_name}--{basis_mode}",
                compression_ratio=compression_ratio,
                approximator_mode=ApproximatorMode.HOMOGENOUS_LOWRANK,
            ),
        )

    for conf in tqdm(arr_experiment_confs):
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

        student, results = distillator.distill(
            student=student,
            approximator=approximator,
            distill_info=distill_info,
            epochs=epochs,
            basis=basis,
            device=device,
            lr=lr,
            log_dir=basis_distillation_output_dir / "log",
            lambda_mse=lambda_mse,
            lambda_xent=lambda_xent,
        )

        min_std = np.min(basis.artifact["std"].numpy())
        max_std = np.max(basis.artifact["std"].numpy())
        print(
            f"std(min)={min_std:.4f} | std(max)={max_std:.4f}",
        )

        utils.dump_json(basis_distillation_output_dir / "results.json", results)

    click.echo(f"Check Results at: {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
