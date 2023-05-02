import click

from tqdm import tqdm
import torchmetrics

from datetime import datetime

from pathlib import Path

import torch
from torch import nn

from xaikd import utils, bases, models

from xaikd import constants
from xaikd.constants import datasets


@torch.no_grad()
def compute_acc(
    model: nn.Module, dataset: datasets.DatasetConfiguration, device: str
) -> float:
    data_loader = dataset.loader(train_split=False)

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=dataset.num_classes)

    for x, y in data_loader:
        logits = model(x.to(device)).cpu()

        metric.update(logits, y)

    metric = metric.compute()

    return float(metric.cpu().detach().numpy())


@click.command()
@click.option("--model-name", type=str)
@click.option("--layer", type=str)
@click.option("--artifact-dir", type=str)
@click.option("--basis-names", default=",".join(constants.BASIS_NAMES))
def main(model_name, layer, basis_names, artifact_dir):
    arguments = locals()

    start_time = datetime.now()

    device = utils.get_device()

    model = models.get_model(model_name).to(device)

    dataset_name = model_name.split("-")[0]

    dataset = datasets.get_constant(dataset_name)

    artifact_dir = Path(artifact_dir) / model_name / layer

    click.echo(f"Loading artifacts from `{artifact_dir}`")
    click.echo(f"Device: {device}")

    assert layer == "layer1"
    # how to get this number?
    dims = 64

    # todo: this has to be part of arch
    module: nn.Module = getattr(model, layer)[-1]

    original_accuracy = compute_acc(model, dataset, device)

    for basis_name in tqdm(
        basis_names.split(","), desc=f"[model={model_name},device={device}]"
    ):
        basis = bases.get_basis(basis_name)

        basis.load(artifact_dir, device=device)

        arr_accs = []
        for k in range(dims):
            try:
                hook = module.register_forward_hook(
                    basis.construct_fh_rank_k_projection(k)
                )
                acc = compute_acc(model, dataset, device)
                arr_accs.append(acc)
            finally:
                hook.remove()

        utils.dump_json(
            f"{artifact_dir}/{basis}/accuracy.json",
            dict(accuracies=acc, dims=dims, original_accuracy=original_accuracy),
        )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
