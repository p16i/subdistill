import click

import os

from tqdm import tqdm
import torchmetrics
import typing

from datetime import datetime

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from xaikd import utils, bases, models

from xaikd import constants
from xaikd import attributors

from xaikd import datasets
from xaikd.utils import metrics
from xaikd.utils.modules import construct_select_logits_of_selected_classes_and_others
import numpy as np


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

    model.to(device)

    dl_train = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )

    dl_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    model_with_selected_logits = DummyModule(
        model=model,
        logit_filter=construct_select_logits_of_selected_classes_and_others(
            selected_classes=dataset.selected_classes,
            total_orig_num_classes=total_orig_num_classes,
        ),
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

        original_accuracy, _ = metrics.accuracy(
            model_with_selected_logits,
            dataloader=dl_val,
            num_classes=dataset.num_classes,
            device=device,
            verbose=True,
        )

        logit_modifier = attributors.WinningClassEvidence(
            num_classes=total_orig_num_classes
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

        _, d = arr_act.shape

        arr_ks = (
            [1] + list(filter(lambda k: k % 2 == 0, np.arange(2, d // 2))) + [d // 2, d]
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
                dataloader=datasets.build_dataloader(
                    dataset.create_subset(train_split=True), shuffle=False
                ),
            )

            arr_accuracies = []
            arr_losses = []
            for k in tqdm(arr_ks, desc=f"[dataset={dataset_name}; basis={basis_name}]"):
                try:

                    hook = module.register_forward_hook(
                        basis.construct_fh_rank_k_projection(k, device=device)
                    )

                    acc, loss = metrics.accuracy(
                        model_with_selected_logits,
                        dl_val,
                        num_classes=dataset.num_classes,
                        device=device,
                        verbose=False,
                    )
                    print(f"basis_name={basis_name}; k={k}: acc={acc}")
                    arr_losses.append(loss)
                    arr_accuracies.append(acc)
                finally:
                    hook.remove()

            os.makedirs(f"{output_dir}/{basis_name}", exist_ok=True)

            utils.dump_json(
                Path(f"{output_dir}/{basis_name}/accuracy.json"),
                dict(
                    accuracies=arr_accuracies,
                    losses=arr_losses,
                    arr_ks=list(map(int, arr_ks)),
                    dims=dims,
                    original_accuracy=original_accuracy,
                ),
            )

    time_took = datetime.now() - start_time
    click.echo(f"Results saved to: {artifact_dir}")
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
