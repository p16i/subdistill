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
from xaikd import datasets, attributors

from xaikd.utils import metrics
import numpy as np

ARR_DIMS = [1, 2, 4, 8, 16, 32, 40, 48, 56]


@click.command()
@click.option("--model-name", type=str)
@click.option("--layer", type=str)
@click.option("--artifact-dir", type=str)
@click.option("--dataset-name", type=str, default="imagenet-cat")
@click.option(
    "--basis-names",
    default="pca-uncentered,prca-sortabs--uncentered,pcalookahead--uncentered",
)
def main(model_name, layer, dataset_name, basis_names, artifact_dir):
    arguments = locals()

    rng = np.random.default_rng(seed=1)

    start_time = datetime.now()

    device = utils.get_device()

    model = models.get_trained_model(model_name)

    _, arch, variant = model_name.split("-")

    dataset = datasets.construct(dataset_name)

    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)
    model.to(device)

    artifact_dir = Path(artifact_dir) / model_name / layer

    dims = constants.ARCH_LAYER_DIMENSIONS[arch][layer]

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

    module: nn.Module = utils.interceptor.get_module(model, layer)

    dl_train = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )

    dl_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    original_accuracy, _ = metrics.accuracy(
        model,
        dataloader=dl_val,
        num_classes=len(dataset.selected_classes),
        device=device,
        verbose=True,
    )

    arr_ks = ARR_DIMS

    logit_modifier = attributors.WinningClassEvidence(
        num_classes=len(dataset.selected_classes)
    )

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
        rng=rng,
        data_loader=dl_train,
        device=device,
        logit_modifier=logit_modifier,
    )

    for basis_name in tqdm(
        basis_names.split(","),
        desc=f"[model={model_name},device={device}]",
    ):
        basis = bases.get_basis(basis_name)

        basis.fit(
            arr_act=arr_act,
            arr_ctx=arr_ctx,
            # this is mainly for pcalookahead
            model=model,
            layer=layer,
            # todo: use shuffle=True
            dataloader=dl_train,
        )

        accuracies = []
        for k in tqdm(arr_ks, desc=f"[dataset={dataset_name}; basis={basis_name}]"):
            try:

                hook = module.register_forward_hook(
                    basis.construct_fh_rank_k_projection(k, device=device)
                )
                acc, _ = metrics.accuracy(
                    model,
                    dl_val,
                    num_classes=dataset.num_classes,
                    device=device,
                    verbose=True,
                )
                accuracies.append(acc)
            finally:
                hook.remove()

        os.makedirs(f"{artifact_dir}/{basis_name}", exist_ok=True)

        utils.dump_json(
            Path(f"{artifact_dir}/{basis_name}/accuracy.json"),
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
