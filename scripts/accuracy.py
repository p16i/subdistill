import click

import os

from tqdm import tqdm
import torchmetrics

from datetime import datetime

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from xaikd import utils, bases, models

from xaikd import constants
from xaikd.constants import datasets


@torch.no_grad()
def compute_acc(
    model: nn.Module, data_loader: DataLoader, num_classes: int, device: str
) -> float:
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    for x, y in data_loader:
        logits = model(x.to(device)).cpu()

        metric.update(logits, y)

    metric = metric.compute()

    return float(metric.cpu().detach().numpy())


@click.command()
@click.option("--model-name", type=str)
@click.option("--layer", type=str)
@click.option("--artifact-dir", type=str)
@click.option(
    "--basis-names", default=",".join(["random1--centered"] + constants.BASIS_NAMES)
)
def main(model_name, layer, basis_names, artifact_dir):
    arguments = locals()

    start_time = datetime.now()

    device = utils.get_device()

    model = models.get_model(model_name).to(device)

    dataset_name, arch, variant = model_name.split("-")

    dataset = datasets.get_constant(dataset_name)

    artifact_dir = Path(artifact_dir) / model_name / layer

    dims = models.get_layer_dimensions(arch, layer)

    click.echo(f"Loading artifacts from `{artifact_dir}`")
    click.echo(f"Device: {device}")
    click.echo(
        " | ".join(
            [
                f"Model: {model_name}",
                f"Layer: {layer} (dims={dims})",
            ]
        )
    )

    # todo: this has to be part of arch
    module: nn.Module = getattr(model, layer)[-1]

    data_loader = dataset.loader(train_split=False, batch_size=128)

    original_accuracy = compute_acc(
        model, data_loader, num_classes=dataset.num_classes, device=device
    )

    arr_ks = list(range(0, dims, 2))

    for basis_name in tqdm(
        basis_names.split(","),
        desc=f"[model={model_name},device={device}]",
    ):
        basis = bases.get_basis(basis_name)

        basis.load(artifact_dir, device=device)

        accuracies = []
        for k in tqdm(arr_ks, desc=f"[basis={basis_name}]"):
            try:
                hook = module.register_forward_hook(
                    basis.construct_fh_rank_k_projection(k)
                )
                acc = compute_acc(
                    model, data_loader, num_classes=dataset.num_classes, device=device
                )
                accuracies.append(acc)
            finally:
                hook.remove()

        os.makedirs(f"{artifact_dir}/{basis}", exist_ok=True)

        utils.dump_json(
            f"{artifact_dir}/{basis}/accuracy.json",
            dict(
                accuracies=accuracies,
                arr_ks=arr_ks,
                dims=dims,
                original_accuracy=original_accuracy,
            ),
        )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
