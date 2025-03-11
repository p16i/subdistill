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
)


from pytorch_lightning.loggers import WandbLogger


WANDB_DIR = os.getenv("WANDB_DIR", ".")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "xaikd-distillation-layerwise-ep3")


def learn_basis(
    teacher_model: nn.Module,
    dataset: datasets.DatasetConfiguration,
    train_loader: DataLoader,
    train_loader_with_shuffle: DataLoader,
    logit_mod: attributors.LogitModifier,
    layers: typing.List[str],
    layer_policy: str,
    device: str,
    output_dir: Path,
    seed: int,
) -> typing.Dict[str, bases.OrthogonalBasis]:
    # prepare bases
    arr_learned_bases = dict()

    if "basis" not in layer_policy:
        return arr_learned_bases

    _, basis_name = layer_policy.split(":")

    rng = np.random.default_rng(seed=seed)

    for layer in layers:
        layer_output_dir = output_dir / f"layer-{layer}"

        os.makedirs(layer_output_dir, exist_ok=True)
        arr_act, arr_ctx = attributors.extract_activation_grad(
            model=teacher_model,
            layer=layer,
            dataloader=train_loader,
            logit_modifier=logit_mod,
            device=device,
            rng=rng,
        )

        click.echo(f"[layer={layer}] fitting basis={basis_name}")
        basis = bases.get_basis(basis_name)
        basis.fit(
            arr_act,
            arr_ctx,
            # for pca-lookahead
            model=teacher_model,
            layer=layer,
            dataloader=train_loader_with_shuffle,
        )

        arr_learned_bases[f"{layer}"] = basis

    return arr_learned_bases


def build_dataloaders(
    dataset: datasets.DatasetConfiguration,
    training_size: float,
    seed: int,
) -> typing.Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    # todo: perhaps, we can abstract this  into datasets and add tests
    # idea: return dl_train, dl_train_aug, dl_val, dl_test (when use_val=False, dl_val = dl_test)
    ds_train = dataset.create_subset(train_split=True)

    # remark: when ratio=1.0, we do this anyway; so the code below is more staight forward
    ds_train = datasets.subsample_dataset(ds_train, ratio=training_size, seed=seed)

    # remark: we have to do it this way because the current version of
    #  `contaminate_dataset` function only work with `Subset.
    ds_val = dataset.create_subset(train_split=False)

    # remark: we set shuffle=False here becaue it is only used to learn bases.
    train_loader = datasets.build_dataloader(ds_train, shuffle=False)
    train_loader_with_shuffle = datasets.build_dataloader(ds_train, shuffle=True)

    val_loader = datasets.build_dataloader(
        ds_val,
        shuffle=False,
    )

    print(f"Dataset Information")
    for label, ds in [("train", ds_train), ("val", ds_val)]:
        print(f"> split={label:5s}: count={len(ds)}")

    ds_train_with_aug = deepcopy(ds_train)

    # We have to make sure that the `dataset` attribute is an actual dataset containing tranform.
    # This avoids having a nested chain of Subsets.
    assert hasattr(ds_train.dataset, "transform")
    assert isinstance(ds_train.dataset, tvd.CIFAR100) or isinstance(
        ds_train.dataset, tvd.ImageNet
    )

    ds_train_with_aug.dataset.transform = dataset.input_training_transformation

    # this loader is used in the distillation process.
    train_loader_with_aug = datasets.build_dataloader(
        ds_train_with_aug,
        shuffle=True,
        # cf. Ahn et al. (2017), VID, in Supplement Sec. A.3.
        # we scale batch_size such that when training_size < 1.0,
        # we get the same number of update steps.
        batch_size=int(np.ceil(64 * training_size)),
        drop_last=True,
    )

    return train_loader, train_loader_with_shuffle, train_loader_with_aug, val_loader


# todo: rename file to distill some-vs-others


@click.command()
@click.option("--teacher", default="cifar100-resnet18-v1", required=True)
@click.option("--student", default="student-32-24-16-8", required=True)
@click.option("--dataset", default="cifar100-people", type=str, required=True)
@click.option("--training-size", type=float, default=1.0, required=True)
@click.option("--layer-policy", type=str, required=True)
@click.option(
    "--last-layer-policy",
    default="binkd",
    type=click.Choice(["binkd", "kd", "dkd"]),
    required=True,
)
@click.option("--layers", default=None, type=str)
@click.option("--lambda-task", default=0.0, type=float)
@click.option("--lambda-kd", default=1.0, type=float)
@click.option("--lambda-layer", default=None, type=float)
@click.option("--default-lambda-layer-config", default=None, type=str)
@click.option("--epochs", type=int, default=100, required=True)
@click.option(
    "--parameter-partition-mode", type=str, default="@0"
)  # todo: what is this again?
@click.option(
    "--finetuning-with-layer-loss", type=bool, default=True
)  # todo: what is this?
@click.option("--lr", type=float, default=0.0005, required=True)
@click.option("--enable-checkpointing", type=bool, default=False, is_flag=True)
@click.option("--wandb-experiment-group", type=str, default=None)
@click.option("--seed", type=int, default=1)
@click.option("--output-dir", type=str, required=True)
def main(
    teacher,
    student,
    dataset,
    training_size,
    last_layer_policy,
    layer_policy,
    layers,
    lambda_task,
    lambda_kd,
    lambda_layer,
    default_lambda_layer_config,
    epochs,
    parameter_partition_mode,  # todo: change to perform-fullupdate
    finetuning_with_layer_loss,
    lr,
    enable_checkpointing,
    seed,
    output_dir,
    wandb_experiment_group,
):

    lambda_layer = constants.resolve_lambda_layer(
        teacher_model_name=teacher,
        policy_name=layer_policy,
        lambda_layer=lambda_layer,
        default_config_key=default_lambda_layer_config,
    )

    arguments = locals()

    pl.seed_everything(seed)

    start_time = datetime.now()

    if layers is None:
        click.echo("layers is not specified. We fall back to the default values")
        layers = constants.DEFAULT_TEACHER_STUDENT_LAYER_MAPPING[teacher]
        click.echo(f"> {layers}")

    arr_teacher_layers, arr_student_layers = distillation_policies.parse_layer_string(
        layers
    )

    output_dir = (
        Path(output_dir)
        / f"{dataset}-tz{training_size}"
        / teacher
        / f"partitionMode{parameter_partition_mode}-seed{seed}"
    )

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    # prepare dataset
    dataset = datasets.construct(dataset)

    train_loader, train_loader_with_shuffle, train_loader_with_aug, val_loader = (
        build_dataloaders(
            dataset,
            training_size=training_size,
            seed=seed,
        )
    )

    # prepare teacher
    layer_logodd_selected_classes = models.layers.LayerLogOddSelectedClasses(
        selected_classes=dataset.selected_classes
    )
    teacher_model = nn.Sequential(
        OrderedDict(
            [
                ("base", models.get_trained_model(teacher).to(device)),
                ("layer_logodd", layer_logodd_selected_classes),
            ]
        )
    )
    teacher_model.eval()
    teacher_model.to(device)

    arr_teacher_layers = list(map(lambda t: f"base.{t}", arr_teacher_layers))

    dict_teacher_layer_dim = utils.get_dimensions_at_layers(
        teacher_model, train_loader, layers=arr_teacher_layers, device=device
    )

    dict_student_layer_dim = utils.get_dimensions_at_layers(
        # we don't this to make sure that we don't use
        models.get_untrained_model(
            student,
            num_classes=dataset.num_classes,
            class_indices=dataset.selected_classes,
        )
        .eval()
        .to(device),
        train_loader,
        layers=arr_student_layers,
        device=device,
    )

    print("Layerwise Distillation with the following layers:")
    for (teacher_layer, teacher_dim), (student_layer, student_dim) in zip(
        dict_teacher_layer_dim.items(),
        dict_student_layer_dim.items(),
    ):
        print(
            f"> mapping `{teacher_layer}` (d={teacher_dim}) to `{student_layer}` (d={student_dim}, parameter_partition_mode={parameter_partition_mode})"
        )

    logit_mod = logit_modifiers.BinaryLogOddWinning(threshold=0)
    arr_learned_bases = learn_basis(
        teacher_model=teacher_model,
        dataset=dataset,
        train_loader=train_loader,
        train_loader_with_shuffle=train_loader_with_shuffle,
        logit_mod=logit_mod,
        layers=arr_teacher_layers,
        layer_policy=layer_policy,
        device=device,
        output_dir=output_dir,
        seed=seed,
    )

    print(f"[policy={layer_policy}] with lambda-layer={lambda_layer}")

    student_model = models.get_untrained_model(
        student, num_classes=dataset.num_classes, class_indices=dataset.selected_classes
    )

    arr_layer_policies = []
    for teacher_layer, student_layer in zip(arr_teacher_layers, arr_student_layers):
        teacher_layer_dims = dict_teacher_layer_dim[teacher_layer]
        student_layer_dims = dict_student_layer_dim[student_layer]

        kwargs = dict(
            teacher_dims=teacher_layer_dims,
            student_dims=student_layer_dims,
            device=device,
        )

        if "basis" in layer_policy:
            # todo: add comment here
            kwargs["basis"] = arr_learned_bases[f"{teacher_layer}"]

            policy_name, _ = layer_policy.split(":")
        else:
            policy_name = layer_policy
        # todo: perhaps, we can just abstract these kwargs into get_layer_policy
        # todo: bring the basis_estimatino here?
        policy = distillation_policies.get_layer_policy(policy_name, **kwargs)

        arr_layer_policies.append(policy)

    distillator = distillators.Layerwise(
        teacher=teacher_model,
        dataset=dataset,
        train_dataloader=train_loader_with_aug,
        val_dataloader=val_loader,
        device=device,
        weight_decay=0.0,  # todo: perhaps, we can just set this to zero in the LayerWise Distillator
        parameter_partition_mode=parameter_partition_mode,
    )

    student_slug = "--".join(
        [
            student,
            layer_policy,
            f"lmd_task{lambda_task}-lmd_kd{lambda_kd}-lmd_layer{lambda_layer}",
        ]
    )

    # todo: what do we save in this dir?
    log_dir = output_dir / "distilled-models" / student_slug
    logger = WandbLogger(
        save_dir=WANDB_DIR,
        project=WANDB_PROJECT,
        group=(
            wandb_experiment_group
            if wandb_experiment_group is not None
            else arguments["output_dir"]
        ),  # todo: simpify this
        job_type="distillation",
        name=f"{student}-{last_layer_policy}-{layer_policy}-seed{seed}",
        notes=f"commit:{utils.get_git_hash()}",
        log_model="all" if enable_checkpointing else False,
        config={
            **arguments,
            "policy": layer_policy,
            "lambda_layer": lambda_layer,
            "output_dir": output_dir,
        },
    )

    trained_student, results = distillator.distill(
        student=student_model,
        last_layer_policy=last_layer_policy,
        layer_policies=distillation_policies.LayerPolicyCollection(
            teacher_layers=arr_teacher_layers,
            student_layers=arr_student_layers,
            policies=arr_layer_policies,
        ),
        epochs=epochs,
        lambda_task=lambda_task,
        lambda_kd=lambda_kd,
        lambda_layer=lambda_layer,
        device=device,
        lr=lr,
        log_dir=log_dir,
        logger=logger,
        seed=seed,
        enable_checkpointing=enable_checkpointing,
        finetuning_with_layer_loss=finetuning_with_layer_loss,
    )

    last_epoch_val_auroc = results["arr_metrics"]["val_auroc"][-1]

    print(f"Result: [distill with:  `{layer_policy}`] auroc={last_epoch_val_auroc:.4f}")

    for k, v in results.items():
        logger.experiment.summary[k] = v


    # do we actually need this?
    # log prediction
    # remark: this prediction is the of the latest model, which is NOT necesseary
    # the best.
    with torch.no_grad():
        teacher_model.to(device)
        trained_student.to(device)

        assert teacher_model.training == trained_student.training == False

        arr_targets = []
        arr_student_pred = []
        arr_teacher_pred = []

        for x, y in val_loader:
            x = x.to(device)
            teacher_pred = teacher_model(x) > 0
            student_logits = trained_student(x)

            assert student_logits.shape == (x.shape[0], 1)

            # todo: log logit instead maybe?
            student_pred = student_logits.squeeze(1) > 0

            arr_targets.extend(y.numpy().tolist())
            arr_teacher_pred.extend(teacher_pred.cpu().numpy().tolist())
            arr_student_pred.extend(student_pred.cpu().numpy().tolist())

        logger.log_table(
            "prediction",
            dataframe=pd.DataFrame.from_dict(
                dict(
                    target=arr_targets,
                    student_pred=arr_student_pred,
                    teacher_pred=arr_teacher_pred,
                )
            ),
        )

    wandb.finish()

    click.echo(f"Check Results at: {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
