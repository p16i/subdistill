import typing
import click
import os
import pandas as pd

import pytorch_lightning as pl
import numpy as np

from datetime import datetime

from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from torchvision import datasets as tvd

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

from xaikd.showcases import cleverhans
from xaikd import distillation_policies
from xaikd.utils import click_types


from pytorch_lightning.loggers import WandbLogger


WANDB_DIR = os.getenv("WANDB_DIR", ".")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "xaikd-distillation-layerwise-ep3")


def learn_basese(
    teacher_model: nn.Module,
    dataset: datasets.DatasetConfiguration,
    train_loader: DataLoader,
    logit_mod: attributors.LogitModifier,
    layers: typing.List[str],
    layer_policies: typing.List[str],
    device: str,
    output_dir: Path,
    seed: int,
) -> typing.Dict[str, bases.Basis]:
    basis_names = list(
        map(
            lambda p: p.split(":")[1],
            filter(lambda p: "basis" in p, layer_policies),
        )
    )
    if len(basis_names) == 0:
        return

    # prepare bases
    arr_learned_bases = dict()

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

        path_mean = layer_output_dir / "act_mean.npy"

        if not os.path.exists(path_mean):
            np.save(path_mean, mean)
        else:
            np.testing.assert_allclose(mean, np.load(path_mean), atol=1e-3)

        print(f"we learn {len(basis_names)} bases: {basis_names}")
        for basis_name in basis_names:
            click.echo(f"[layer={layer}] fitting basis={basis_name}")
            basis = bases.get_basis(basis_name, seed=seed)
            basis.fit(
                arr_act,
                arr_ctx,
                mean=mean,
                device=device,
                model=teacher_model,
                layer=layer,
                dataloader=train_loader,
            )
            basis.save(layer_output_dir)

            arr_learned_bases[f"{layer}-{basis_name}"] = basis

    return arr_learned_bases


def build_dataloaders(
    dataset: datasets.DatasetConfiguration,
    training_size: float,
    contamination_level: float,
    seed: int,
    use_val_split: bool,
) -> typing.Tuple[DataLoader, DataLoader, DataLoader]:
    if use_val_split:
        assert training_size == 1.0

        # We set this because we also scale batch-size rather.
        training_size = 0.8
        ds_train, ds_val = random_split(
            dataset.create_subset(train_split=True),
            [training_size, 1 - training_size],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        ds_train = datasets.subsample_dataset(
            dataset.create_subset(train_split=True), ratio=training_size, seed=seed
        )
        # remark: we have to do it this way because the current version of
        #  `contaminate_dataset` function only work with `Subset.
        ds_val = datasets.subsample_dataset(
            dataset=dataset.create_subset(train_split=False), ratio=1.0, seed=1
        )

    if contamination_level > 0:
        ds_train = cleverhans.contaminate_dataset(
            ds_train,
            contamination_level=contamination_level,
            seed=seed,
            victim_class_indices=[min(dataset.selected_classes)],
        )

        ds_val = cleverhans.contaminate_dataset(
            dataset=ds_val,
            contamination_level=contamination_level,
            seed=seed,
            # remark: here, we assume that, in the validation data for distillation,
            # all validaiton samples of only one class has spuriour correlation.
            victim_class_indices=dataset.selected_classes,
        )

    # remark: we set shuffle=False here becaue it is only used to learn bases.
    train_loader = datasets.build_dataloader(ds_train, shuffle=False)
    val_loader = datasets.build_dataloader(
        ds_val,
        shuffle=False,
    )

    print(f"Dataset Information: [use_val_split={use_val_split}]")
    for label, dl in [("train", train_loader), ("val", val_loader)]:
        count = 0
        for _, y in dl:
            count += y.shape[0]

        print(f"> split={label:5s}: count={count}")

    # We have to make sure that the `dataset` attribute is an actual dataset containing tranform.
    # This avoids having a nested chain of Subsets.
    assert isinstance(ds_train.dataset, tvd.CIFAR100) or isinstance(
        ds_train.dataset, tvd.ImageNet
    )

    ds_train_with_aug = deepcopy(ds_train)
    ds_train_with_aug.dataset.transform = dataset.input_training_transformation

    # this loader is used in the distillation process.
    train_loader_with_aug = datasets.build_dataloader(
        ds_train_with_aug,
        shuffle=True,
        # cf. Ahn et al. (2017), VID, in Supplement Sec. A.3.
        # we scale batch_size such that when training_size < 1.0,
        # we get the same number of update steps.
        batch_size=int(np.ceil(64 * training_size)),
    )

    return train_loader, train_loader_with_aug, val_loader


@click.command()
@click.option("--dataset", default="cifar100-people", type=str, required=True)
@click.option("--teacher", default="cifar100-resnet18-v1", required=True)
@click.option("--student", default="resnet18xscifarcompr1", required=True)
@click.option(
    "--layers", default="layer1,layer2,layer3,layer4", type=str, required=True
)
@click.option(
    "--layer-policies",
    type=str,
    default="basis-identity:pca--uncentered,basis-identity:prca-sortabs--uncentered,basis-identity:random--uncentered,attention-transfer,vid,fitnet,nothing",
    required=True,
)
@click.option("--output-dir", type=str, required=True)
@click.option("--training-size", type=float, default=0.1, required=True)
@click.option("--epochs", type=int, default=100, required=True)
@click.option("--seed", type=int, default=1)
@click.option("--lr", type=float, default=0.0005, required=True)
@click.option("--lambda-task", default=0.0, type=float)
@click.option("--lambda-kd", default=1.0, type=float)
@click.option("--lambda-layer", type=float, default=None)
@click.option("--contamination-level", default=0.0, type=float)
@click.option("--use-val-split", type=bool, default=False, is_flag=True)
@click.option("--enable-checkpointing", type=bool, default=False, is_flag=True)
@click.option("--detach-layer-output", type=bool, default=False)
@click.option(
    "--learning-bases-from-clean-data", type=bool, default=False, is_flag=True
)
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
    lambda_task,
    lambda_kd,
    lambda_layer,
    contamination_level,
    use_val_split,
    learning_bases_from_clean_data,
    enable_checkpointing,
    detach_layer_output,
):
    arguments = locals()

    pl.seed_everything(seed)

    start_time = datetime.now()

    teacher_layers, student_layers = distillation_policies.parse_layer_string(layers)

    layer_policies = layer_policies.split(",")

    output_dir = (
        Path(output_dir)
        / f"{dataset}-clv{contamination_level}-tz{training_size}-valsplit{use_val_split}-cleanDSBasis{learning_bases_from_clean_data}-detachLayerOutput{detach_layer_output}-seed{seed}"
        / teacher
    )

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    # prepare dataset
    dataset = datasets.construct(dataset)

    logit_mod = attributors.WinningClassEvidence(num_classes=dataset.num_classes)

    train_loader, train_loader_with_aug, val_loader = build_dataloaders(
        dataset,
        training_size=training_size,
        contamination_level=contamination_level,
        seed=seed,
        use_val_split=use_val_split,
    )

    # prepare teacher
    teacher_model = models.get_trained_model(teacher)
    # use only teacher's logits corresponding to selected classes
    utils.modify_last_layer_for_subclasses(
        teacher_model, selected_classes=dataset.selected_classes
    )
    teacher_layer_dims_mapping = utils.get_dimensions_at_layers(
        teacher_model, train_loader, layers=teacher_layers
    )
    teacher_model.to(device)

    student_layer_dims_mapping = utils.get_dimensions_at_layers(
        models.get_untrained_model(student, num_classes=dataset.num_classes).eval(),
        train_loader,
        layers=student_layers,
    )

    print("Layerwise Distillation with the following layers:")
    for (teacher_layer, teacher_dim), (student_layer, student_dim) in zip(
        teacher_layer_dims_mapping.items(),
        student_layer_dims_mapping.items(),
    ):
        print(
            f"> mapping `{teacher_layer}` (d={teacher_dim}) to `{student_layer}` (d={student_dim}, detach_layer_output={detach_layer_output})"
        )

    if learning_bases_from_clean_data:
        assert not use_val_split

        train_loader_for_learning_bases = datasets.build_dataloader(
            dataset.create_subset(train_split=True), shuffle=False
        )
    else:
        train_loader_for_learning_bases = train_loader

    arr_learned_bases = learn_basese(
        teacher_model=teacher_model,
        dataset=dataset,
        train_loader=train_loader_for_learning_bases,
        logit_mod=logit_mod,
        layers=teacher_layers,
        layer_policies=layer_policies,
        device=device,
        output_dir=output_dir,
        seed=seed,
    )

    # do distillation
    for policy_name_with_args in tqdm(layer_policies, desc="Distillation"):
        # this make sure that we use the same initial student model for all policy.
        pl.seed_everything(seed)

        policy_slugs = policy_name_with_args.split(":")

        print(f"[policy={policy_name_with_args}]")
        if lambda_layer is None:
            policy_lambda_layer = constants.get_lamba_layer_for_policy_student(
                policy_name_with_args, student
            )
            print(f"> lambda_layer={policy_lambda_layer} (specified via `constants`)")
        else:
            policy_lambda_layer = lambda_layer
            print(
                f"[> lambda_layer={policy_lambda_layer} (specified via `command line`)"
            )

        policy_name = policy_slugs[0]

        student_model = models.get_untrained_model(
            student, num_classes=dataset.num_classes
        )

        layer_policies = []
        for teacher_layer, student_layer in zip(teacher_layers, student_layers):
            teacher_layer_dims = teacher_layer_dims_mapping[teacher_layer]
            student_layer_dims = student_layer_dims_mapping[student_layer]

            kwargs = dict(
                teacher_dims=teacher_layer_dims,
                student_dims=student_layer_dims,
                device=device,
            )

            if "basis" in policy_name:
                basis_name = policy_slugs[-1]

                if basis_name == "pcalookahead--uncentered":
                    print(">>>> pcalookadhead <<<<")
                    basis = arr_learned_bases[f"{layer}-{basis_name}"]
                else:
                    basis = bases.get_basis(basis_name, seed=seed)
                    layer_output_dir = output_dir / f"layer-{teacher_layer}"
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
            detach_layer_output_in_forward_hook=detach_layer_output,
        )

        student_slug = "--".join(
            [
                student,
                "-".join(policy_slugs),
                f"lmd_task{lambda_task}-lmd_kd{lambda_kd}-lmd_layer{policy_lambda_layer}",
            ]
        )

        log_dir = output_dir / "distilled-models" / student_slug
        logger = WandbLogger(
            save_dir=WANDB_DIR,
            project=WANDB_PROJECT,
            group=arguments["output_dir"],
            job_type="distillation",
            name=f"{student}-{policy_name_with_args}-seed{seed}",
            notes=f"commit:{utils.get_git_hash()}",
            log_model="all" if enable_checkpointing else False,
            config={
                **arguments,
                "policy": policy_name_with_args,
                "policy_lambda_layer": policy_lambda_layer,
                "output_dir": output_dir,
            },
        )

        trained_student, results = distillator.distill(
            student=student_model,
            layer_policies=distillation_policies.LayerPolicyCollection(
                teacher_layers=teacher_layers,
                student_layers=student_layers,
                policies=layer_policies,
            ),
            epochs=epochs,
            lambda_task=lambda_task,
            lambda_kd=lambda_kd,
            lambda_layer=policy_lambda_layer,
            device=device,
            lr=lr,
            log_dir=log_dir,
            logger=logger,
            seed=seed,
            enable_checkpointing=enable_checkpointing,
        )

        # todo: save student to artifacts!

        last_epoch_val_acc = results["arr_metrics"]["val_acc"][-1]
        last_epoch_val_agreement = results["arr_metrics"]["val_agreement"][-1]

        print(
            f"Result: [distill with:  `{policy_name_with_args}`] acc={last_epoch_val_acc:.4f} agreement={last_epoch_val_agreement:.4f}"
        )

        for k, v in results.items():
            logger.experiment.summary[k] = v

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
