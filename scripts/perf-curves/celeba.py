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
@click.option("--arch", default="celeba-resnet18-pretrained", type=str)
@click.option(
    "--arr-layers",
    default=["layer1", "layer2", "layer3", "layer4"],
    required=True,
    type=click_types.List(),
)
@click.option("--task-id", required=True, type=int)
# todo: we need to implement this
# @click.option("--sample-selection-criteria", type=str)
@click.option("--output-dir", default="./tmp", type=click_types.Path())
@click.option("--num-steps", default=20)
@click.option("--max-k", default=None, type=int)
@click.option("--seed", default=1)
@click.option("--data-size", default=1.0)
@click.option(
    "--logodd-threshold",
    default=0,
)
@click.option(
    "--arr-basis-names",
    default=["pca", "gradpca", "prcaposdef", "prcasortabs", "prca"],
    type=click_types.List(),
)
def main(
    arch: str,
    arr_layers: click_types.List.output_type,
    task_id: int,
    arr_basis_names: click_types.List.output_type,
    # sample_selection_criteria: str,
    data_size: int,
    output_dir: click_types.Path.output_type,
    num_steps: int,
    logodd_threshold: float,
    seed: int,
    max_k: typing.Union[None, int],
):
    arguments = locals()
    start_time = datetime.now()

    rng = np.random.default_rng(seed=seed)
    device = utils.get_device()

    dataset = datasets.construct("celeba")

    dataset_name = f"celeba-task{task_id}"

    layer_task_logit_selection = models.layers.TaskLogitSelection(task_id=task_id)

    base_model = models.get_trained_model(arch)

    model = nn.Sequential(
        OrderedDict(
            [
                ("base", base_model),
                ("layer_task_logit", layer_task_logit_selection),
            ]
        )
    )

    model.eval()
    model.to(device)

    logit_modifier = logit_modifiers.BinaryLogOddWinning(threshold=logodd_threshold)

    ds_train = datasets.subsample_dataset(
        dataset=dataset.create_subset(train_split=True), ratio=data_size, seed=seed
    )

    click.echo(
        f"Perf Curve for `{dataset_name}` (data_size={data_size}, logit_modifier={logit_modifier})"
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

    metric = metrics.MetricAUROC()

    (ref_auroc,) = metric(model=model, dataloader=dl_val, device=device, verbose=True)
    print(f"Ref metric={metric} (val_set)={ref_auroc:.4f}")

    for layer in tqdm(arr_layers, desc="estimate performance curve at layer"):
        d = dict_layer_dims[layer]
        stop_at = d if max_k is None else max_k

        arr_ks = np.linspace(start=1, stop=stop_at, num=num_steps).astype(int)
        # we include K=d for sanity check
        arr_ks = list(sorted(set(arr_ks.tolist() + [d])))

        base_layer_name = f"base.{layer}"

        arr_act, arr_grad = attributors.extract_activation_grad(
            model=model,
            layer=base_layer_name,
            dataloader=dl_train,
            logit_modifier=logit_modifier,
            rng=rng,
            device=device,
        )

        for basis_name in tqdm(
            arr_basis_names,
            desc=f"[dataset_name={dataset_name},layer={layer},d={d}] arr_ks={arr_ks}",
        ):
            basis = bases.get_basis(basis_name=basis_name)

            basis.fit(arr_act=arr_act, arr_ctx=arr_grad)

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
                    (_auroc,) = (
                        interceptor.attach_projection_forward_hook_at_layer_and_evaluate_metrics(
                            model=model,
                            layer=base_layer_name,
                            dataloader=dl,
                            forward_hook_func=forward_hook_func,
                            metric=metric,
                            device=device,
                        )
                    )

                    dict_stats[f"{prefix}_auroc"] = _auroc

                arr_stat_rows.append(dict_stats)

            df = pd.DataFrame(arr_stat_rows)

            # todo: parameterize also `sample-selection-criteria`
            dest_path = output_dir / arch / dataset_name / layer / basis_name
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
