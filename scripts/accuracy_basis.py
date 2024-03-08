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
    seed: int,
):
    rng = np.random.default_rng(seed=seed)
    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
        data_loader=loader,
        logit_modifier=logit_modifier,
        device=device,
        rng=rng,
    )

    print("arr_act.shape", arr_act.shape)

    mean_act = np.mean(arr_act, axis=0)
    np.save(output_dir / "act_mean", mean_act)

    for basis_name in basis_names:
        click.echo(f"Learning {basis_name}")

        basis = bases.get_basis(basis_name, seed=seed)

        basis.fit(arr_act, arr_ctx, mean=mean_act, device=device)

        basis.save(output_dir)

    return arr_act, arr_ctx


@torch.no_grad()
def estimate_acc_for_basis(
    model: nn.Module,
    layer: str,
    dataset: datasets.Cifar100SuperClassesDataset,
    dataloader: DataLoader,
    basis: bases.Basis,
    device: str,
    arr_ks: typing.List[int],
) -> typing.Tuple[typing.List[float], typing.List[float]]:
    module = utils.interceptor.get_module(model, layer)

    arr_accs = []
    arr_xent = []

    for k in tqdm(arr_ks, desc=f"[layer={layer}: basis={basis}]"):
        try:
            hook = module.register_forward_hook(
                basis.construct_fh_rank_k_projection(k, device)
            )

            acc, xent = metrics.accuracy(
                model,
                dataloader=dataloader,
                num_classes=dataset.num_classes,
                device=device,
            )

        finally:
            hook.remove()

        arr_accs.append(acc)
        arr_xent.append(xent)

    return arr_accs, arr_xent


@click.command()
@click.option("--model", type=click_types.Model(), required=True)
@click.option("--dataset", type=str, required=True)
@click.option(
    "--layers",
    type=click_types.List(),
    default="layer1,layer2,layer3,layer4",
)
@click.option("--output-dir", default=Path("./tmp"), type=click_types.Path())
@click.option(
    "--basis-mode", default="uncentered", type=click.Choice(["centered", "uncentered"])
)
@click.option(
    "--basis-names",
    type=click_types.List(),
    default="pca,prca-sortabs,random",
)
@click.option("--seed", default=1, type=int)
@click.option(
    "--logit-mod", default="winning", type=click.Choice(["winning", "target"])
)
@click.option("--training-size", default=1.0, type=float)
def main(
    model: nn.Module,
    dataset: str,
    layers: typing.List[str],
    output_dir: Path,
    seed: int,
    basis_mode: str,
    basis_names: typing.List[str],
    training_size: float,
    logit_mod,
):
    pl.seed_everything(seed)
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(dataset)

    model = model.to(device)
    utils.modify_last_layer_for_subclasses(
        model, selected_classes=dataset.selected_classes
    )

    train_ds = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=training_size, seed=seed
    )

    train_dataloader = datasets.build_dataloader(train_ds, shuffle=False)

    val_dataloader = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False, batch_size=128
    )

    click.echo(f"Basis Centering Mode: {basis_mode}")
    click.echo(f"with bases: {basis_names}")

    if logit_mod == "winning":
        logit_modifier = attributors.WinningClassEvidence(
            num_classes=dataset.num_classes
        )
    elif logit_mod == "target":
        logit_modifier = attributors.TargetClassEvidence(
            num_classes=dataset.num_classes
        )
    else:
        raise

    original_acc, original_xent = metrics.accuracy(
        model,
        dataloader=val_dataloader,
        num_classes=dataset.num_classes,
        device=device,
    )

    dataset_slug = getattr(dataset, "__name")
    dataset_slug = f"{dataset_slug}--ts{training_size}-seed{seed}"

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

        arr_act, arr_ctx = extract_activation_and_bases(
            model=model,
            dataset=dataset,
            loader=train_dataloader,
            output_dir=layer_output_dir,
            basis_names=basis_names_with_mode,
            layer=layer,
            device=device,
            logit_modifier=logit_modifier,
            seed=seed,
        )

        dims = models.get_layer_output_dimensions(model, layer)
        arr_ks = (
            np.unique(np.array(utils.logspace(dims) + list(range(1, 31 + 1))))
            .astype(int)
            .tolist()
        )

        print(f"Computing with arr_ks={arr_ks}")

        for basis_name in basis_names_with_mode:
            basis = bases.get_basis(basis_name, seed=seed)

            basis.load(layer_output_dir, device=device)

            arr_acc, arr_xent = estimate_acc_for_basis(
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
                    arr_xent=arr_xent,
                    arr_ks=arr_ks,
                    arr_compressions=(dims / np.array(arr_ks)).astype(float).tolist(),
                    unexplained_relevance=metrics.unexplained_relevance(
                        arr_act, arr_ctx, basis.artifact["eigvecs"].cpu().numpy()
                    ),
                    dims=dims,
                    original_acc=original_acc,
                    original_xent=original_xent,
                    num_classes=dataset.num_classes,
                ),
            )

    utils.dump_json_with_string_serializer(output_dir / "meta.json", arguments)
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
