import click
import typing
import os

from tqdm import tqdm

from datetime import datetime

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from xaikd import utils, bases, models

from xaikd import constants
from xaikd import attributors
from xaikd.utils.modules import construct_select_logits_of_selected_classes_and_others

from xaikd import datasets
from xaikd.utils import metrics
import numpy as np
import pandas as pd


class DummyModule(nn.Module):
    def __init__(self, model: nn.Module, logit_filter: typing.Callable):
        super().__init__()
        self.model = model
        self.logit_filter = logit_filter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logit_filter(self.model(x))


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

    total_orig_num_classes: int = model.__last_layer.weight.shape[0]
    print(f"using device={device} (with n={torch.cuda.device_count()})")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    ds_train = dataset.create_subset(train_split=True)

    trng = torch.Generator()
    trng.manual_seed(1)
    ds_train_small, _ = random_split(ds_train, [0.1, 0.9], generator=trng)

    num_workers = utils.get_num_workers()
    click.echo(f"Using {num_workers} workers!")

    dl_train = DataLoader(
        ds_train,
        batch_size=64,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    dl_train_small = DataLoader(
        ds_train_small,
        batch_size=64,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    dl_val = DataLoader(
        dataset.create_subset(train_split=False),
        batch_size=64,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
    )

    logit_filters = construct_select_logits_of_selected_classes_and_others(
        dataset.selected_classes,
        total_orig_num_classes=total_orig_num_classes,
    )

    model_with_modified_logits = DummyModule(model, logit_filters)

    with torch.no_grad():
        orig_accuracy, orig_xent, orig_arr_aurocs = metrics.accuracy(
            model=model_with_modified_logits,
            dataloader=dl_val,
            num_classes=dataset.num_classes,
            device=device,
            verbose=True,
        )

    ref_stats = dict(
        orig_loss=orig_xent,
        orig_accuracy=orig_accuracy,
    )

    for cix, auroc in enumerate(orig_arr_aurocs):
        ref_stats[f"orig_auroc_c{cix}"] = auroc

    logit_modifier = attributors.WinningClassEvidence(
        num_classes=total_orig_num_classes
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

        arr_ks = np.linspace(1, d, num=10).astype(int)

        for basis_name in tqdm(
            basis_names.split(","),
            desc=f"[model={model_name},device={device}]",
        ):
            basis = bases.get_basis(basis_name)

            basis.fit(
                arr_act=arr_act,
                arr_ctx=arr_ctx,
            )

            arr_rows = []
            for k in tqdm(arr_ks, desc=f"[dataset={dataset_name}; basis={basis_name}]"):
                row = dict(
                    k=k,
                    **ref_stats,
                )

                projector = basis.construct_fh_rank_k_projection(k, device=device)

                for dataset_label, dl in [
                    ("train", dl_train),
                    ("val", dl_val),
                ]:
                    hook = None
                    try:
                        hook = module.register_forward_hook(projector)
                        with torch.no_grad():
                            acc, loss, arr_aurocs = metrics.accuracy(
                                model_with_modified_logits,
                                dl,
                                num_classes=dataset.num_classes,
                                device=device,
                                verbose=True,
                            )

                        row[f"{dataset_label}_loss"] = loss
                        row[f"{dataset_label}_acc"] = acc

                        for cix, auroc in enumerate(arr_aurocs):
                            row[f"{dataset_label}_auroc_c{cix}"] = auroc
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
