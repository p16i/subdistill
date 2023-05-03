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

from xaikd.utils import click_types


def extract_activation_and_bases(
    model: nn.Module,
    dataset: datasets.DatasetConfiguration,
    output_dir: Path,
    basis_names: typing.List[str],
    layer: str,
    device: str,
):
    logit_modifier = attributors.OneClassEvidence(dataset)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
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


def estimate_auroc(
    model: nn.Module,
    dataloader: DataLoader,
    logodd_mod: attributors.LogOddEvidence,
    device: str,
) -> float:
    class1, class2 = logodd_mod.classes

    metric = BinaryAUROC()
    for x, y in dataloader:
        logits = model(x.to(device))

        logodd = logodd_mod(logits).sum(dim=1).detach().cpu()

        assert np.logical_or(y == class1, y == class2).all()

        ybin = np.where(y == class1, 0, 1)
        metric.update(logodd, torch.from_numpy(ybin))

    auroc = metric.compute()

    return float(auroc)


@torch.no_grad()
def estimate_auroc_for_basis(
    model: nn.Module,
    layer: str,
    dataloader: DataLoader,
    logodd_mod: attributors.LogOddEvidence,
    basis: bases.Basis,
    device: str,
    arr_ks: typing.List[int],
) -> typing.List[float]:
    # todo: this has to be attract in the model/arch logic!
    module: nn.Module = getattr(model, layer)[-1]

    arr_aurocs = []

    for k in tqdm(arr_ks, desc=f"[layer={layer}: basis={basis}]"):
        try:
            hook = module.register_forward_hook(basis.construct_fh_rank_k_projection(k))

            auroc = estimate_auroc(
                model,
                dataloader=dataloader,
                logodd_mod=logodd_mod,
                device=device,
            )

        finally:
            hook.remove()

        arr_aurocs.append(auroc)

    return arr_aurocs


@click.command()
@click.option("--model", type=click_types.Model(), required=True)
@click.option("--dataset", type=click_types.DatasetConfiguration(), required=True)
@click.option("--layers", default=",".join(["layer1", "layer2", "layer3", "layer4"]))
@click.option("--output-dir", default=Path("./tmp"), type=click_types.Path())
@click.option("--seed", default=1, type=int)
def main(
    model: nn.Module,
    dataset: datasets.TwoclassesDataset,
    layers: str,
    output_dir: Path,
    seed: int,
):
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    layers: typing.List[str] = layers.split(",")

    model = model.to(device)

    val_dataloader = dataset.loader(train_split=False)

    logodd_mod = attributors.LogOddEvidence(dataset.selected_classes, dataset)

    original_auroc = estimate_auroc(model, val_dataloader, logodd_mod, device)

    for layer in layers:
        layer_output_dir = (
            Path(output_dir)
            / getattr(dataset, "__name")
            / getattr(model, "__name")
            / layer
        )
        os.makedirs(layer_output_dir, exist_ok=True)

        # remark: should we set seed globally or every layer?
        np.random.seed(seed)

        extract_activation_and_bases(
            model=model,
            dataset=dataset,
            output_dir=layer_output_dir,
            basis_names=constants.BASIS_NAMES,
            layer=layer,
            device=device,
        )

        dims = models.get_layer_dimensions(model, layer)
        arr_ks = list(range(0, dims + 1, 2))

        for basis_name in constants.BASIS_NAMES:
            basis = bases.get_basis(basis_name)

            basis.load(layer_output_dir, device=device)

            arr_auroc = estimate_auroc_for_basis(
                model=model,
                layer=layer,
                dataloader=val_dataloader,
                logodd_mod=logodd_mod,
                basis=basis,
                device=device,
                arr_ks=arr_ks,
            )

            utils.dump_json(
                layer_output_dir / f"{basis}" / "stats.json",
                dict(
                    arr_auroc=arr_auroc,
                    arr_ks=arr_ks,
                    dims=dims,
                    original_auroc=original_auroc,
                ),
            )

    utils.dump_json_with_string_serializer(output_dir / "meta.json", arguments)
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
