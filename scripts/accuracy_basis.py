import os
import typing
import click
import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from datetime import datetime

from pathlib import Path

from xaikd import utils, datasets, attributors, bases, constants, models

from xaikd.utils import click_types, metrics

import pytorch_lightning as pl


def extract_activation_and_bases(
    model: nn.Module,
    dataset: datasets.Cifar100SuperClassesDataset,
    loader: DataLoader,
    output_dir: Path,
    basis_names: typing.List[str],
    layer: str,
    device: str,
    logit_modifier: attributors.LogitModifier,
):
    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
        data_loader=loader,
        logit_modifier=logit_modifier,
        device=device,
    )

    print("arr_act.shape", arr_act.shape)

    mean_act = np.mean(arr_act, axis=0)
    np.save(output_dir / "act_mean", mean_act)

    for basis_name in basis_names:
        click.echo(f"Learning {basis_name}")

        basis = bases.get_basis(basis_name)

        basis.fit(arr_act, arr_ctx, mean=mean_act, device=device)

        basis.save(output_dir)


@torch.no_grad()
def estimate_acc_for_basis(
    model: nn.Module,
    layer: str,
    dataset: datasets.Cifar100SuperClassesDataset,
    dataloader: DataLoader,
    basis: bases.Basis,
    device: str,
    arr_ks: typing.List[int],
) -> typing.List[float]:
    module = utils.interceptor.get_module(model, layer)

    arr_accs = []

    for k in tqdm(arr_ks, desc=f"[layer={layer}: basis={basis}]"):
        try:
            hook = module.register_forward_hook(basis.construct_fh_rank_k_projection(k))

            acc = metrics.accuracy_with_subclasses(
                model,
                dataloader,
                dataset.selected_classes,
                dataset.transform_target,
                device=device,
            )

        finally:
            hook.remove()

        arr_accs.append(acc)

    return arr_accs


@click.command()
@click.option("--model", type=click_types.Model(), required=True)
@click.option("--dataset", type=str, required=True)
@click.option(
    "--layers",
    type=click_types.List(),
    default="layer3,layer4",
)
@click.option("--output-dir", default=Path("./tmp"), type=click_types.Path())
@click.option(
    "--basis-mode", default="centered", type=click.Choice(["centered", "uncentered"])
)
@click.option(
    "--logit-modifier",
    default="oneclass",
    type=click.Choice(["oneclass", "multipleclasses"]),
)
@click.option(
    "--basis-names",
    type=click_types.List(),
    default="pca,prca-abs,random1,random2",
)
@click.option("--seed", default=1, type=int)
@click.option("--num-training-samples", default=50, type=int)
def main(
    model: nn.Module,
    dataset: str,
    layers: typing.List[str],
    output_dir: Path,
    seed: int,
    basis_mode: str,
    basis_names: typing.List[str],
    num_training_samples: typing.Union[None, int],
    logit_modifier: str,
):
    pl.seed_everything(seed)
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    model = model.to(device)

    # todo: seed should be here?

    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        dataset, num_training_samples=num_training_samples
    )
    train_dataloader = dataset.loader(train_split=True)

    # remark: we need to use `batch_size=1` due to rounding issue.
    val_dataloader = dataset.loader(train_split=False, batch_size=1)

    click.echo(f"Basis Centering Mode: {basis_mode}")
    click.echo(f"with bases: {basis_names}")

    if logit_modifier == "oneclass":
        logit_mod = attributors.OneClassEvidence(dataset)
    elif logit_modifier == "multipleclasses":
        logit_mod = attributors.SelectedClassesEvidence(dataset)
    else:
        raise ValueError("")

    original_acc = metrics.accuracy_with_subclasses(
        model,
        val_dataloader,
        dataset.selected_classes,
        dataset.transform_target,
        device=device,
    )

    dataset_slug = getattr(dataset, "__name")
    if num_training_samples is not None:
        dataset_slug = f"{dataset_slug}--n{num_training_samples}"

    output_dir = (
        Path(output_dir)
        / dataset_slug
        / f"logit-mod-{logit_modifier}"
        / getattr(model, "__name")
    )

    click.echo(f"Output: {output_dir}")

    for layer in layers:
        pl.seed_everything(seed)

        layer_output_dir = output_dir / layer
        os.makedirs(layer_output_dir, exist_ok=True)

        basis_names_with_mode = list(map(lambda s: f"{s}--{basis_mode}", basis_names))

        non_random_bases = list(
            filter(lambda s: not "random" in s, basis_names_with_mode)
        )

        extract_activation_and_bases(
            model=model,
            dataset=dataset,
            loader=train_dataloader,
            output_dir=layer_output_dir,
            basis_names=non_random_bases,
            layer=layer,
            device=device,
            logit_modifier=logit_mod,
        )

        dims = models.get_layer_dimensions(model, layer)
        arr_ks = [0, 1, 2, 3] + list(range(4, dims + 2, 4))

        for basis_name in basis_names_with_mode:
            basis = bases.get_basis(basis_name)

            basis.load(layer_output_dir, device=device)

            arr_acc = estimate_acc_for_basis(
                model=model,
                layer=layer,
                dataset=dataset,
                dataloader=val_dataloader,
                basis=basis,
                device=device,
                arr_ks=arr_ks,
            )

            basis_output_dir = layer_output_dir / f"{basis}"
            os.makedirs(basis_output_dir, exist_ok=True)

            utils.dump_json(
                basis_output_dir / "stats.json",
                dict(
                    arr_acc=arr_acc,
                    arr_ks=arr_ks,
                    dims=dims,
                    original_auroc=original_acc,
                ),
            )

    utils.dump_json_with_string_serializer(output_dir / "meta.json", arguments)
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
