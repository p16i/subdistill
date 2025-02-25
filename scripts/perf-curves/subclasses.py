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

from xaikd import (
    utils,
    models,
    datasets,
    attributors,
    bases,
    metrics,
    interceptor,
    logit_modifiers,
)
from xaikd.utils import click_types


@click.command()
@click.option("--arch", default="cifar100-resnet18-v1", type=str)
@click.option(
    "--arr-layers",
    default=["layer1", "layer2", "layer3", "layer4"],
    required=True,
    type=click_types.List(),
)
@click.option("--dataset-name", required=True, type=str)
@click.option("--logit-modifier", default="MultiClassLogOddWinning", type=str)
# @click.option("--sample-selection-criteria", type=str)
@click.option("--output-dir", default="./tmp", type=click_types.Path())
@click.option("--seed", default=1)
@click.option("--data-size", default=1.0)
@click.option(
    "--arr-basis-names",
    default=["pca", "gradpca", "prcaposdef", "prca-ablation-a-c"],
    type=click_types.List(),
)
def main(
    arch: str,
    arr_layers: click_types.List.output_type,
    dataset_name: str,
    arr_basis_names: click_types.List.output_type,
    # sample_selection_criteria: str,
    logit_modifier: str,
    data_size: int,
    output_dir: click_types.Path.output_type,
    seed: int,
):
    arguments = locals()
    start_time = datetime.now()

    rng = np.random.default_rng(seed=seed)
    device = utils.get_device()

    dataset = datasets.construct(dataset_name)

    base_model = models.get_trained_model(arch)

    model = nn.Sequential(
        OrderedDict(
            [
                ("base", base_model),
                (
                    "subclass_selection",
                    models.layers.SubclassSelection(dataset.selected_classes),
                ),
            ]
        )
    )

    model.eval()
    model.to(device)

    logit_modifier_obj = logit_modifiers.get_logit_modifier(name=logit_modifier)

    ds_train = datasets.subsample_dataset(
        dataset=dataset.create_subset(train_split=True), ratio=data_size, seed=seed
    )

    click.echo(
        f"Perf Curve for `{dataset_name}` (data_size={data_size}, logit_modifier={logit_modifier_obj})"
    )

    dl_train = datasets.build_dataloader(
        ds_train,
        shuffle=False,
    )
    dl_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False),
        shuffle=False,
    )

    dict_layer_dims = utils.get_dimensions_at_layers(
        model=base_model, dataloader=dl_train, layers=arr_layers, device=device
    )

    metric = metrics.MetricAccuracy(num_classes=len(dataset.selected_classes))

    (ref_acc,) = metric(model=model, dataloader=dl_val, device=device, verbose=True)
    print(f"Ref metric={metric} (val_set)={ref_acc:.4f}")

    for layer in tqdm(arr_layers, desc="estimate performance curve at layer"):
        d = dict_layer_dims[layer]
        stop_at = np.floor(np.log2(d))

        arr_ks = np.power(2, np.arange(0, stop_at))
        arr_ks = np.unique(arr_ks.tolist() + [d]).astype(int)

        base_layer_name = f"base.{layer}"

        arr_act, arr_grad = attributors.extract_activation_grad(
            model=model,
            layer=base_layer_name,
            dataloader=dl_train,
            logit_modifier=logit_modifier_obj,
            rng=rng,
            device=device,
        )

        for basis_name in tqdm(
            arr_basis_names, desc=f"[layer={layer},d={d}] arr_ks={arr_ks}"
        ):
            basis = bases.get_basis(basis_name=basis_name)

            basis.fit(arr_act=arr_act, arr_ctx=arr_grad)

            arr_stat_rows = []

            for k in tqdm(arr_ks, desc=f"basis_name={basis_name}"):

                forward_hook_func = basis.construct_fh_rank_k_projection(
                    k=k, device=device
                )

                dict_stats = {
                    "arch": arch,
                    "layer": layer,
                    "d": d,
                    "dataset_name": dataset_name,
                    "basis_name": basis_name,
                    "k": k,
                    f"ref_{metric}": ref_acc,
                }

                for prefix, dl in [("val", dl_val)]:
                    (_acc,) = (
                        interceptor.attach_projection_forward_hook_at_layer_and_evaluate_metrics(
                            model=model,
                            layer=base_layer_name,
                            dataloader=dl,
                            forward_hook_func=forward_hook_func,
                            metric=metric,
                            device=device,
                        )
                    )

                    dict_stats[f"{prefix}_{metric}"] = _acc

                arr_stat_rows.append(dict_stats)

            df = pd.DataFrame(arr_stat_rows)

            # todo: parameterize also `sample-selection-criteria`
            dest_path = (
                output_dir
                / arch
                / dataset_name
                / layer
                / f"data-size{data_size}"
                / basis_name
                / logit_modifier
            )
            os.makedirs(dest_path, exist_ok=True)

            df.to_csv(dest_path / "stats.csv", index=False)
            utils.dump_json_with_string_serializer(
                dest=dest_path / "meta.json", data=arguments
            )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")
    click.echo(f"check results at {output_dir}")


if __name__ == "__main__":
    main()
