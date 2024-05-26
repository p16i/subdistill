import re
import typing
import os
import click
from pathlib import Path

from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np
import numpy.typing as npt
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F

from xaikd import models, datasets, utils, attributors, constants, bases, constants

from xaikd.utils import metrics


DEVICE = utils.get_device()


@click.command()
@click.option("--dataset-name", type=str)
@click.option("--model-name", type=str)
@click.option("--basis-names", type=str, default="pca,prca-sortabs,pcalookahead")
@click.option(
    "--basis-mode",
    type=click.Choice(["centered", "uncentered"]),
    default="uncentered",
)
@click.option("--output-dir", type=str)
def main(model_name, dataset_name, output_dir, basis_names, basis_mode):
    arguments = locals()
    start_time = datetime.now()

    _, arch, _ = model_name.split("-")
    arr_layers = list(constants.ARCH_LAYER_DIMENSIONS[arch].keys())

    click.echo(f"> dataset={dataset_name}")
    click.echo(f"> model={model_name}, layers={arr_layers}")

    output_path = Path(output_dir) / dataset_name / model_name

    arr_basis_names = basis_names.split(",")

    dataset = datasets.construct(dataset_name)

    model = models.get_trained_model(model_name)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)

    model.to(DEVICE)

    logit_modifier = attributors.WinningClassEvidence(
        num_classes=len(dataset.selected_classes)
    )

    print(f"LogitMod: {logit_modifier}")

    dataloader_train = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )
    dataloader_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    ref_acc, _ = metrics.accuracy(
        model,
        dataloader=dataloader_val,
        num_classes=dataset.num_classes,
        device=DEVICE,
    )
    print(f"> ref_acc={ref_acc}")

    rng = np.random.default_rng(seed=1)

    for basis_name in tqdm(arr_basis_names):

        basis_output_path = output_path / f"{basis_name}--{basis_mode}"
        os.makedirs(basis_output_path, exist_ok=True)

        basis_ref_acc, _ = metrics.accuracy(
            model,
            dataloader=dataloader_val,
            num_classes=dataset.num_classes,
            device=DEVICE,
        )

        np.testing.assert_allclose(
            basis_ref_acc, ref_acc, err_msg="sanity check: ref acc remains the same"
        )

        arr_statistics = []

        arr_layer_bases: list[bases.Basis] = []
        for layer in arr_layers:
            arr_act, arr_ctx = attributors.extract_activation_context(
                model=model,
                layer=layer,
                dataset=dataset,
                rng=rng,
                data_loader=dataloader_train,
                device=DEVICE,
                logit_modifier=logit_modifier,
            )
            layer_basis = bases.get_basis(f"{basis_name}--{basis_mode}")
            layer_basis.fit(
                arr_act=arr_act,
                arr_ctx=arr_ctx,
                # this is mainly for pcalookahead
                model=model,
                layer=layer,
                dataloader=dataloader_train,
            )

            arr_layer_bases.append(layer_basis)

        for layer_dimensions in constants.ARR_STUDENT_DIMENSIONS:
            assert len(layer_dimensions) == len(arr_layers)

            arr_hooks = []
            try:
                for layer, layer_basis, k in zip(
                    arr_layers, arr_layer_bases, layer_dimensions
                ):
                    module = utils.interceptor.get_module(model=model, layer_str=layer)

                    hook = module.register_forward_hook(
                        layer_basis.construct_fh_rank_k_projection(k, device=DEVICE)
                    )
                    arr_hooks.append(hook)

                compressed_acc, _ = metrics.accuracy(
                    model,
                    dataloader=dataloader_val,
                    num_classes=dataset.num_classes,
                    device=DEVICE,
                )

                arr_statistics.append(
                    dict(
                        basis_name=basis_name,
                        layer_dimensions=layer_dimensions,
                        ref_acc=ref_acc,
                        acc=compressed_acc,
                    )
                )

            finally:
                for hook in arr_hooks:
                    hook.remove()

        pd.DataFrame(arr_statistics).to_csv(
            basis_output_path / f"stats.csv", index=False
        )

    click.echo(f"Check output at: {output_path}")
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
