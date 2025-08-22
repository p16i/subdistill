import typing
import click
import os
import pandas as pd

from collections import OrderedDict

import pytorch_lightning as pl
import numpy as np

from datetime import datetime

from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import datasets as tvd

import wandb

from xaikd import (
    attributors,
    bases,
    constants,
    datasets,
    distillation_policies,
    distillators,
    logit_modifiers,
    models,
    utils,
    metrics,
)


from pytorch_lightning.loggers.wandb import WandbLogger


TEACHER_LAYER_PREFIX = "base"


WANDB_ENTITY = os.getenv("WANDB_ENTITY", "xaikd")
WANDB_DIR = os.getenv("WANDB_DIR", "/tmp")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "test")


@click.command()
@click.option("--teacher", default="cifar100-resnet18-v1", required=True)
@click.option("--student", default="student-32-24-16-8", required=True)
@click.option("--dataset", default="cifar100-people", type=str, required=True)
@click.option("--training-size", type=float, default=1.0, required=True)
@click.option("--distillation-policy", type=str, required=True)
@click.option("--layers", default=None, type=str)
@click.option("--lambda-layer", default=None, type=float)
@click.option("--default-lambda-layer-config", default=None, type=str)
@click.option("--batch-size", default=64)
@click.option("--epochs", type=int, default=100, required=True)
@click.option("--lr", type=float, default=0.0005, required=True)
@click.option("--weight-decay", type=float, default=0, required=True)
@click.option("--layerwise-training", type=bool, default=False)
@click.option("--upload-best-checkpoint", type=bool, default=False, is_flag=True)
@click.option("--wandb-experiment-group", type=str, default=None)
@click.option("--seed", type=int, default=1)
@click.option("--output-dir", type=str, default="/tmp")
def main(
    teacher,
    student,
    dataset,
    training_size,
    distillation_policy,
    layers,
    lambda_layer,
    default_lambda_layer_config,
    epochs,
    lr,
    weight_decay,
    upload_best_checkpoint,
    seed,
    output_dir,
    wandb_experiment_group,
    batch_size,
    layerwise_training,
):
    (
        lambda_collection,
        layer_policy,
    ) = distillation_policies.resolve_lambdas_and_layer_policy(
        teacher=teacher,
        policy_name=distillation_policy,
        lambda_layer=lambda_layer,
        default_lambda_layer_config=default_lambda_layer_config,
        layerwise_training=layerwise_training,
    )

    wanddb_experiment_group = (
        wandb_experiment_group if not wandb_experiment_group is None else output_dir
    )

    arguments = locals()

    pl.seed_everything(seed)

    start_time = datetime.now()

    arr_teacher_layers, arr_student_layers = distillation_policies.parse_layer_string(
        layers
    )

    output_dir = (
        Path(output_dir) / f"{dataset}-tz{training_size}" / teacher / f"seed{seed}"
    )

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    # prepare dataset
    dataset = datasets.construct(dataset)

    (
        train_loader,
        train_loader_with_aug,
        val_loader,
        test_loader,
    ) = datasets.construct_dataloaders(
        dataset=dataset,
        training_data_ratio=training_size,
        seed=seed,
        use_validation_set=True,
        training_batch_size=batch_size,
    )

    # prepare teacher
    teacher_model = nn.Sequential(
        OrderedDict(
            [
                (TEACHER_LAYER_PREFIX, models.get_trained_model(teacher).to(device)),
                (
                    "last_layer",
                    models.layers.resolve_teacher_last_layer(dataset=dataset),
                ),
            ]
        )
    )
    teacher_model.eval()
    teacher_model.to(device)

    arr_teacher_layers = list(
        map(lambda t: f"{TEACHER_LAYER_PREFIX}.{t}", arr_teacher_layers)
    )

    dict_teacher_layer_dim = utils.get_dimensions_at_layers(
        teacher_model, train_loader, layers=arr_teacher_layers, device=device
    )

    student_model = models.get_untrained_model(
        student,
        num_classes=dataset.num_classes,
    ).to(device)

    dict_student_layer_dim = utils.get_dimensions_at_layers(
        deepcopy(student_model).eval(),
        train_loader,
        layers=arr_student_layers,
        device=device,
    )

    logit_mod = logit_modifiers.BinaryLogOddWinning(threshold=0)

    print(
        f"[distillation_policy={distillation_policy} layer_policy={layer_policy}] with {lambda_collection}"
    )

    if len(arr_teacher_layers) > 0:
        print(f"We attach `layer_policy={layer_policy}` at the following layers:")

    for (teacher_layer, teacher_dim), (student_layer, student_dim) in zip(
        dict_teacher_layer_dim.items(),
        dict_student_layer_dim.items(),
    ):
        print(
            f"> mapping `{teacher_layer}` (d={teacher_dim}) to `{student_layer}` (d={student_dim})"
        )

    arr_layer_policies = []
    for teacher_layer, student_layer in zip(arr_teacher_layers, arr_student_layers):
        teacher_layer_dims = dict_teacher_layer_dim[teacher_layer]
        student_layer_dims = dict_student_layer_dim[student_layer]

        kwargs = dict(
            teacher_dims=teacher_layer_dims,
            student_dims=student_layer_dims,
        )

        if "basis" in layer_policy:
            policy_name, basis_slug = layer_policy.split(":")
            basis_name = bases.resolve_basis_name_for_layer(
                slug=basis_slug,
                layer=teacher_layer.replace(f"{TEACHER_LAYER_PREFIX}.", ""),
            )
            basis = bases.helpers.learn_basis(
                teacher_model=teacher_model,
                train_loader=train_loader,
                logit_mod=logit_mod,
                layer=teacher_layer,
                basis_name=basis_name,
                device=device,
                seed=seed,
            )

            auroc_at_k = bases.helpers.evaluate_basis_at_k(
                teacher_model=teacher_model,
                basis=basis,
                layer=teacher_layer,
                metric_func=metrics.MetricAUROCBinaryCrossEntropy(),
                train_loader=None,
                val_loader=val_loader,
                arr_ks=[dict_student_layer_dim[student_layer]],
                device=device,
            )["val_auroc"].values[0]

            print(f"basis evaluation at k: auroc={auroc_at_k}")

            arguments[f"basis_{student_layer}@k"] = auroc_at_k

            policy = distillation_policies.get_policy(
                policy_name,
                device=device,
                basis=basis,
                layerwise_training=layerwise_training,
                **kwargs,
            )
            if hasattr(policy, "scaling_factor"):
                print(f"> scaling factor: {policy.scaling_factor}")

        else:
            policy = distillation_policies.get_policy(
                layer_policy, device=device, **kwargs
            )

        arr_layer_policies.append(policy)

    # fixme: transfer last layer to student mode

    if arr_student_layers[-1] == "layer4":
        print("transfer last layer")
        policy = arr_layer_policies[-1]

        assert isinstance(
            policy, distillation_policies.OrthogonalBasisCenterRotationV2Policy
        )

        W_teacher = getattr(teacher_model, TEACHER_LAYER_PREFIX).fc.weight
        b_teacher = getattr(teacher_model, TEACHER_LAYER_PREFIX).fc.bias

        k = dict_student_layer_dim["layer4"]
        Uk = torch.from_numpy(policy.basis.get_Uk(k=k)).float().to(device)
        mean = torch.from_numpy(policy.basis.mean).float().to(device)
        new_weight = W_teacher @ Uk
        new_bias = b_teacher - W_teacher @ (Uk.T @ mean)

        student.fc.weight.data = new_weight.data
        student.fc.bias.data = new_bias.data

        utils.freeze_model(student.fc)

    distillator = distillators.Layerwise(
        teacher=teacher_model,
        dataset=dataset,
        dataloader_train=train_loader_with_aug,
        dataloader_val=val_loader,
        dataloader_test=test_loader,
        device=device,
    )

    logger = WandbLogger(
        entity=WANDB_ENTITY,
        save_dir=WANDB_DIR,
        project=WANDB_PROJECT,
        group=wanddb_experiment_group,
        job_type="distillation",
        name=f"{student}-{distillation_policy}-seed{seed}",
        notes=f"commit:{utils.get_git_hash()}",
        config={
            **arguments,
            "distillation_policy": distillation_policy,
            "layer_policy": layer_policy,
            "lambda_layer": lambda_collection.lambda_layer,
            "output_dir": output_dir,
        },
    )

    distillator.distill(
        student=student_model,
        last_layer_policy=distillation_policies.kd.BinaryKLPolicy(device=device),
        layer_policies=distillation_policies.interface.LayerPolicyCollection(
            teacher_layers=arr_teacher_layers,
            student_layers=arr_student_layers,
            policies=arr_layer_policies,
        ),
        epochs=epochs,
        lambda_task=lambda_collection.lambda_task,
        lambda_kd=lambda_collection.lambda_kd,
        lambda_layer=lambda_collection.lambda_layer,
        layerwise_training=layerwise_training,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        logger=logger,
        seed=seed,
        upload_best_checkpoint=upload_best_checkpoint,
    )

    wandb.finish()

    click.echo(f"Check Results at: {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
