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
from xaikd.utils import click_types


from pytorch_lightning.loggers import WandbLogger


WANDB_PROJECT = os.getenv("WANDB_PROJECT", "xaikd-distillation-layerwise")


BATCHSIZE = 128


@click.command()
@click.option("--dataset", default="cifar100-people", type=str, required=True)
@click.option("--teacher", default="cifar100-resnet18-wb15", required=True)
@click.option("--student", default="resnet18compr2", required=True)
@click.option("--layers", default="layer3,layer4", type=str, required=True)
@click.option(
    "--layer-policies",
    type=str,
    default="basis:pca,basis:prca-sortabs,basis:random,vid,linstudent,linteacher",
    required=True,
)
@click.option("--basis-mode", type=str, default="centered", required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--training-size", type=float, default=1.0, required=True)
@click.option("--epochs", type=int, default=200, required=True)
@click.option("--seed", type=int, default=1)
@click.option("--lr", type=float, default=0.0005, required=True)
@click.option("--lambda-kd", type=click_types.SmartFloat(), default=1.0)
@click.option("--lambda-task", type=float, default=0.0)
@click.option("--lambda-layer", type=click_types.SmartFloat(), required=True)
@click.option("--skip-if-exist", type=bool, default=False, is_flag=True)
@click.option("--skip-baselines", type=bool, default=False, is_flag=True)
def main(
    teacher,
    student,
    dataset,
    layer_policies,
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
    layer_policies = layer_policies.split(",")

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
        ds_train_with_aug,
        shuffle=True,
        # why do we scale batch_size like this?
        batch_size=int(np.ceil(BATCHSIZE * training_size)),
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

        basis_names = list(
            map(
                lambda p: p.split(":")[1],
                filter(lambda p: "basis" in p, layer_policies),
            )
        )

        print(f"we learn {len(basis_names)} bases: {basis_names}")

        for basis_name in basis_names:
            click.echo(f"[layer={layer}] fitting basis={basis_name}--{basis_mode}")
            basis = bases.get_basis(f"{basis_name}--{basis_mode}", seed=seed)
            basis.fit(arr_act, arr_ctx, mean=mean, device=device)
            basis.save(layer_output_dir)

    # do distillation
    for policy_name_with_args in tqdm(layer_policies, desc="Distillation"):
        # this make sure that we use the same initial student model for all policy.
        pl.seed_everything(seed)

        policy_slugs = policy_name_with_args.split(":")

        policy_name = policy_slugs[0]

        student_model = models.get_untrained_model(
            student, num_classes=dataset.num_classes
        )

        layer_policies = []
        for layer in layers:
            student_layer_dims = constants.ARCH_LAYER_DIMENSIONS[student][layer]
            teacher_layer_dims = models.get_layer_output_dimensions(
                teacher_model, layer
            )

            kwargs = dict(
                teacher_dims=teacher_layer_dims,
                student_dims=student_layer_dims,
                device=device,
            )

            if policy_name == "basis":
                basis_name = policy_slugs[-1]
                basis = bases.get_basis(f"{basis_name}--{basis_mode}", seed=seed)
                layer_output_dir = output_dir / f"layer-{layer}"
                basis.load(layer_output_dir)

                kwargs["basis"] = basis

            policy = distillation_policies.get_layer_policy(policy_name, **kwargs)

            layer_policies.append(policy)

        distillator = distillators.Layerwise(
            teacher=teacher_model,
            dataset=dataset,
            train_dataloader=train_loader_with_aug,
            val_dataloader=val_loader,
            device=device,
            weight_decay=0.0,
        )

        student_slug = "--".join(
            [
                student,
                "-".join(policy_slugs),
                f"lmd_task{lambda_task}-lmd_kd{lambda_kd}-lmd_layer{lambda_layer}",
            ]
        )

        log_dir = output_dir / "distilled-models" / student_slug
        logger = WandbLogger(
            project=WANDB_PROJECT,
            group=arguments["output_dir"],
            job_type="distillation",
            name=f"{student}-{policy_name_with_args}-seed{seed}",
            config={
                **arguments,
                "policy": policy_name_with_args,
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
            seed=seed,
        )

        # todo: save student to artifacts!

        last_epoch_val_acc = results["arr_metrics"]["val_acc"][-1]
        last_epoch_val_agreement = results["arr_metrics"]["val_agreement"][-1]

        print(
            f"Result: [distill with:  `{policy_name}`] acc={last_epoch_val_acc:.4f} agreement={last_epoch_val_agreement:.4f}"
        )

        for k, v in results.items():
            logger.experiment.summary[k] = v

        # log prediction
        with torch.no_grad():
            teacher_model.to(device)
            trained_student.to(device)

            arr_targets = []
            arr_student_pred = []
            arr_teacher_pred = []

            for x, y in val_loader:
                x = x.to(device)
                teacher_pred = torch.argmax(teacher_model(x), dim=1).cpu()
                student_pred = torch.argmax(trained_student(x), dim=1).cpu()
                arr_targets.extend(y.numpy().tolist())
                arr_teacher_pred.extend(teacher_pred.numpy().tolist())
                arr_student_pred.extend(student_pred.numpy().tolist())

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
