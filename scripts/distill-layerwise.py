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
    constants,
)

from xaikd import distillation_policies


from pytorch_lightning.loggers import WandbLogger


WANDB_PROJECT = os.getenv("WANDB_PROJECT", "xaikd-distillation-layerwise")


@click.command()
@click.option("--dataset", default="cifar100-people", type=str, required=True)
@click.option("--teacher", default="cifar100-resnet18-p1", required=True)
@click.option("--student", default="resnet18compr2", required=True)
@click.option("--layers", default="layer3,layer4", type=str, required=True)
@click.option(
    "--basis-names",
    type=str,
    default="pca,prca-recon,prca-sortabs",
    required=True,
)
@click.option("--basis-mode", type=str, default="centered", required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--training-size", type=float, default=1.0, required=True)
@click.option("--epochs", type=int, default=100, required=True)
@click.option("--seed", type=int, default=1)
@click.option("--lr", type=float, default=0.0005, required=True)
@click.option("--lambda-layer", type=float, default=1.0)
@click.option("--lambda-kd", type=float, default=1.0)
@click.option("--lambda-task", type=float, default=0.0)
@click.option("--skip-if-exist", type=bool, default=False, is_flag=True)
@click.option("--skip-baselines", type=bool, default=False, is_flag=True)
def main(
    teacher,
    student,
    dataset,
    basis_names,
    output_dir,
    seed,
    epochs,
    lr,
    training_size,
    layers,
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

    layers = layers.split(",")
    basis_names = basis_names.split(",")

    output_dir = Path(output_dir) / f"{dataset}-tz{training_size}-seed{seed}" / teacher

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    # prepare dataset
    dataset = datasets.construct(dataset)

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

    # prepare teacher
    teacher_model = models.get_trained_model(teacher)
    # use only teacher's logits corresponding to selected classes
    utils.modify_last_layer_for_subclasses(
        teacher_model.fc, selected_classes=dataset.selected_classes
    )
    teacher_model.to(device)

    # prepare bases
    for layer in layers:
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

    # do distillation
    for basis_name in tqdm(basis_names + ["learnlin"], desc="Distillation"):
        pl.seed_everything(seed)

        student_model = models.get_untrained_model(
            student, num_classes=dataset.num_classes
        )

        layer_policies = []
        for layer in layers:
            dim = constants.ARCH_LAYER_DIMENSIONS[student][layer]

            # todo: refactor to remove this if-else condition
            if basis_name == "learnlin":
                layer_policies.append(
                    distillation_policies.LearnableAdapterPolicy(
                        teacher_dims=models.get_layer_output_dimensions(
                            teacher_model, layer
                        ),
                        student_dims=dim,
                        device=device,
                    ),
                )
            else:
                basis = bases.get_basis(f"{basis_name}--{basis_mode}", seed=seed)
                layer_output_dir = output_dir / f"layer-{layer}"
                basis.load(layer_output_dir)

                layer_policies.append(
                    distillation_policies.OrthogonalBasisPolicy(basis, dim, device),
                )

        distillator = distillators.Layerwise(
            teacher=teacher_model,
            dataset=dataset,
            train_dataloader=train_loader_with_aug,
            val_dataloader=val_loader,
            device=device,
            weight_decay=0.0,
        )

        log_dir = output_dir / "distilled-models" / student
        logger = WandbLogger(
            project=WANDB_PROJECT,
            group=arguments["output_dir"],
            job_type="distillation",
            name=f"{student}-{basis_name}-seed{seed}",
            config={
                **arguments,
                "basis_name": basis_name,
                "output_dir": output_dir,
            },
        )

        trained_student, results = distillator.distill(
            student=student_model,
            layer_policies=distillation_policies.LayerPolicyCollection(
                layers=layers, policies=layer_policies
            ),
            epochs=epochs,
            lambda_task=lambda_task,
            lambda_kd=lambda_kd,
            lambda_layer=lambda_layer,
            device=device,
            lr=lr,
            log_dir=log_dir,
            logger=logger,
        )

        # todo: save student to artifacts!

        last_epoch_val_acc = results["arr_metrics"]["val_acc"][-1]
        last_epoch_val_agreement = results["arr_metrics"]["val_agreement"][-1]

        print(
            f"Result: [distill with:  `{basis_name}`] acc={last_epoch_val_acc:.4f} agreement={last_epoch_val_agreement:.4f}"
        )

        for k, v in results.items():
            logger.experiment.summary[k] = v

        wandb.finish()

    click.echo(f"Check Results at: {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
