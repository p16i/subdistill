import click

import os

from tqdm import tqdm
import torchmetrics

from datetime import datetime

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from xaikd import utils, bases, models

from xaikd import constants
from xaikd import attributors

from xaikd import datasets
from xaikd.utils import metrics
import numpy as np
import pandas as pd


@click.command()
@click.option("--model-name", type=str)
@click.option("--layers", type=str)
@click.option("--dataset-name", type=str, default="imagenet-butterfly")
@click.option(
    "--basis-names",
    default="pca--uncentered,prca-sortabs--uncentered",
)
@click.option("--artifact-dir", type=str, default="/tmp")
def main(model_name, layers, dataset_name, basis_names, artifact_dir):
    arguments = locals()

    rng = np.random.default_rng(seed=1)

    start_time = datetime.now()

    device = utils.get_device()

    model = models.get_trained_model(model_name)

    dataset = datasets.construct(dataset_name)

    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)
    print(f"using device={device} (with n={torch.cuda.device_count()})")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    ds_train = dataset.create_subset(train_split=True)

    trng = torch.Generator()
    trng.manual_seed(1)
    ds_train_small, _ = random_split(ds_train, [0.1, 0.9], generator=trng)

    dl_train = DataLoader(
        ds_train,
        batch_size=64,
        num_workers=16,
        pin_memory=True,
        shuffle=False,
    )
    dl_train_small = DataLoader(
        ds_train_small,
        batch_size=64,
        num_workers=16,
        pin_memory=True,
        shuffle=False,
    )

    dl_val = DataLoader(
        dataset.create_subset(train_split=False),
        batch_size=64,
        num_workers=16,
        pin_memory=True,
        shuffle=False,
    )

    original_accuracy, original_xent = metrics.accuracy(
        model,
        dataloader=dl_val,
        num_classes=len(dataset.selected_classes),
        device=device,
        verbose=True,
    )

    for layer in layers.split(","):

        output_dir = Path(artifact_dir) / dataset_name / model_name / layer

        dims = utils.get_dimensions_at_layers(
            model=model, dataloader=dl_train, layers=[layer], device=device
        )[layer]

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

        logit_modifier = attributors.WinningClassEvidence(
            num_classes=len(dataset.selected_classes)
        )

        arr_act, arr_ctx = attributors.extract_activation_context(
            model=model,
            layer=layer,
            dataset=dataset,
            rng=rng,
            data_loader=dl_train_small,
            device=device,
            logit_modifier=logit_modifier,
        )

        _, d = arr_act.shape

        arr_ks = np.linspace(1, d, num=16).astype(int)

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
                dataloader=datasets.build_dataloader(
                    dataset.create_subset(train_split=True), shuffle=False
                ),
            )

            arr_rows = []
            for k in tqdm(arr_ks, desc=f"[dataset={dataset_name}; basis={basis_name}]"):
                row = dict(
                    k=k,
                    original_loss=original_xent,
                    original_accuracy=original_accuracy,
                )

                projector = basis.construct_fh_rank_k_projection(k, device=device)

                for dataset_label, dl in [
                    ("train", dl_train),
                    ("val", dl_val),
                ]:
                    hook = None
                    try:
                        hook = module.register_forward_hook(projector)
                        acc, loss = metrics.accuracy(
                            model,
                            dl,
                            num_classes=dataset.num_classes,
                            device=device,
                            verbose=True,
                        )

                        row[f"{dataset_label}_loss"] = loss
                        row[f"{dataset_label}_acc"] = acc
                    finally:
                        if hook is not None:
                            hook.remove()
                arr_rows.append(row)

            os.makedirs(f"{output_dir}/{basis_name}", exist_ok=True)

            df = pd.DataFrame(arr_rows)
            df.to_csv(
                Path(f"{output_dir}/{basis_name}/accuracy.csv"),
                index=False,
            )

    time_took = datetime.now() - start_time
    click.echo(f"Results saved to: {artifact_dir}")
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
