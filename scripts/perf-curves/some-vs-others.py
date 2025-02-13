import typing
import click
import os


from collections import OrderedDict
from pathlib import Path

from datetime import datetime


from torch import nn
import torch
from torch.utils.data import random_split
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from xaikd import utils, models, datasets, attributors, bases
from xaikd.utils import click_types, metrics


@click.command()
@click.option("--arch", default="cifar100-resnet18-v1", type=str)
@click.option(
    "--arr-layers", default=["layer1", "layer2"], required=True, type=click_types.List()
)
@click.option("--dataset-name", required=True, type=str)
# @click.option("--sample-selection-criteria", type=str)
@click.option("--output-dir", default="./tmp", type=click_types.Path())
@click.option("--num-steps", default=20)
@click.option("--seed", default=1)
@click.option("--data-size", default=1.0)
@click.option(
    "--logodd-threshold",
    default=0,
)
@click.option(
    "--arr-basis-names",
    default=["pca", "gradpca"],
    type=click_types.List(),
)
def main(
    arch: str,
    arr_layers: click_types.List.output_type,
    dataset_name: str,
    arr_basis_names: click_types.List.output_type,
    # sample_selection_criteria: str,
    data_size: int,
    output_dir: click_types.Path.output_type,
    num_steps: int,
    logodd_threshold: float,
    seed: int,
):
    arguments = locals()
    start_time = datetime.now()

    rng = np.random.default_rng(seed=seed)
    device = utils.get_device()

    dataset = datasets.construct(dataset_name)

    layer_logodd_selected_classes = models.layers.LayerLogOddSelectedClasses(
        selected_classes=dataset.selected_classes
    )

    base_model = models.get_trained_model(arch)

    model = nn.Sequential(
        OrderedDict(
            [
                ("base", base_model),
                ("layer_logodd", layer_logodd_selected_classes),
            ]
        )
    )

    model.eval()
    model.to(device)

    logit_modifier = attributors.BinaryLogOddWinning(threshold=logodd_threshold)

    ds_train = dataset.create_subset(train_split=True)

    trng = torch.Generator()
    trng.manual_seed(seed)
    if data_size < 1.0:
        ds_train, _ = random_split(ds_train, [data_size, 1 - data_size], generator=trng)

    click.echo(
        f"Perf Curve for `{dataset_name}` (data_size={data_size}, logit_modifier={logit_modifier})"
    )

    dl_train = datasets.build_dataloader(
        ds_train,
        shuffle=False,
    )
    ds_val = dataset.create_subset(train_split=False)
    dl_val = datasets.build_dataloader(
        ds_val,
        shuffle=False,
    )

    dict_layer_dims = utils.get_dimensions_at_layers(
        model=base_model, dataloader=dl_train, layers=arr_layers, device=device
    )

    metric = metrics.MetricAUROC()

    ref_auroc = metric(model=model, dataloader=dl_val, device=device, verbose=True)[
        "auroc"
    ]
    print(f"Ref auroc (val)={ref_auroc:.4f}")

    for layer in tqdm(arr_layers, desc="estimate performance curve at layer"):
        d = dict_layer_dims[layer]
        arr_ks = np.linspace(start=1, stop=d, num=num_steps).astype(int)

        base_layer_name = f"base.{layer}"

        arr_act, arr_grad = attributors.extract_activation_grad(
            model=model,
            layer=base_layer_name,
            dataloader=dl_train,
            logit_modifier=logit_modifier,
            rng=rng,
            device=device,
        )

        for basis_name in tqdm(arr_basis_names, desc=f"[layer={layer}]"):
            basis = bases.get_basis(basis_name=basis_name)

            basis.fit(arr_act=arr_act, arr_ctx=arr_grad)

            # todo: parameterize also `sample-selection-criteria`
            artifact_dir = output_dir / arch / dataset_name / layer
            os.makedirs(artifact_dir, exist_ok=True)

            arr_stat_rows = []

            for k in tqdm(arr_ks, desc=f"basis_name={basis_name}"):

                forward_hook_func = basis.construct_fh_rank_k_projection(
                    k=k, device=device
                )

                dict_stats = dict(
                    arch=arch,
                    layer=layer,
                    d=d,
                    dataset_name=dataset_name,
                    basis_name=basis_name,
                    k=k,
                    ref_auroc=ref_auroc,
                )

                for prefix, dl in [("val", dl_val)]:
                    _stats = utils.interceptor.attach_projection_forward_hook_at_layer_and_evaluate_metrics(
                        model=model,
                        layer=base_layer_name,
                        dataloader=dl_val,
                        forward_hook_func=forward_hook_func,
                        metric=metric,
                        device=device,
                    )

                    for k, v in _stats.items():
                        dict_stats[f"{prefix}-{k}"] = v

                arr_stat_rows.append(dict_stats)

            df = pd.DataFrame(arr_stat_rows)

            dest_file = artifact_dir / f"{basis_name}.csv"

            df.to_csv(dest_file, index=False)

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")
    click.echo(f"check results at {output_dir}")


if __name__ == "__main__":
    main()
