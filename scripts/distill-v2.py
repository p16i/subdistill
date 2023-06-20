import click
import os
import pandas as pd

import pytorch_lightning as pl
import numpy as np
import torch

from datetime import datetime

from pathlib import Path
from copy import deepcopy

from torchvision import transforms

from xaikd.utils import click_types
from xaikd import datasets, utils, distillators, models, attributors, bases

from tensorboard_logger import configure


@click.command()
@click.option("--dataset", default="cifar100-35vs98", type=str, required=True)
@click.option("--model", default="cifar100-resnet18-p1", required=True)
@click.option("--layer", default="layer3", type=str, required=True)
@click.option(
    "--basis-names",
    type=str,
    default="pca,prca-abs,random1,random2,random3",
    required=True,
)
@click.option("--basis-mode", type=str, default="centered", required=True)
@click.option("--compression-rate", type=float, default=0.25, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--num-samples", type=int, default=100, required=True)
@click.option("--epochs", type=int, default=50, required=True)
@click.option("--lr", type=float, default=0.001, required=True)
@click.option("--seed", type=int, default=1)
def main(
    model,
    dataset,
    basis_names,
    output_dir,
    compression_rate,
    seed,
    epochs,
    lr,
    num_samples,
    layer,
    basis_mode,
):
    pl.seed_everything(seed)

    arguments = locals()
    start_time = datetime.now()

    model = models.get_model(model)
    model_name = getattr(model, "__name")

    layer_slug = f"layer{layer}-n{num_samples}-comp{compression_rate}-seed{seed}"

    output_dir = Path(output_dir) / dataset / model_name / layer_slug

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    dataset: datasets.TwoClassesDataset = datasets.construct(
        dataset, num_training_samples=num_samples
    )

    model.to(device)

    logodd_mod = attributors.LogOddEvidence(dataset.selected_classes)

    train_loader = dataset.loader(train_split=True, shuffle=True)
    val_loader = dataset.loader(train_split=False, shuffle=False)

    train_loader_with_aug = deepcopy(train_loader)
    # todo: convert this to utils
    train_loader_with_aug.dataset.dataset.transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            train_loader_with_aug.dataset.dataset.transform,
        ]
    )

    # todo: make sure that all bases use the same activation and context vectors
    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        data_loader=train_loader,
        dataset=dataset,
        logit_modifier=logodd_mod,
        device=device,
    )
    mean = np.mean(arr_act, axis=0)
    np.save(output_dir / "act_mean", mean)

    distill_info = distillators.get_distill_infor(
        arch=model_name, layer=layer, compression_rate=compression_rate
    )

    layer_approximator = distillators.get_approximator_for_resnet18(
        layer,
        distill_info.num_output_channels,
    )

    distillator = distillators.Layerwise(
        teacher=model,
        dataset=dataset,
        compression_rate=compression_rate,
        device=device,
    )

    for basis_name in basis_names.split(","):
        basis_name = f"{basis_name}--{basis_mode}"
        basis_output_dir = output_dir / basis_name
        os.makedirs(basis_output_dir, exist_ok=True)

        basis = bases.get_basis(basis_name)

        basis.fit(arr_act, arr_ctx, mean=mean, device=device)
        basis.save(output_dir)

        basis.load(output_dir)

        student = models.get_model(model_name)

        results = distillator.distill(
            student=student,
            approx_mod=deepcopy(layer_approximator),
            distill_info=distill_info,
            epochs=epochs,
            train_dataloader=train_loader_with_aug,
            val_dataloader=val_loader,
            basis=basis,
            seed=seed,
            device=device,
            lr=lr,
            log_dir=basis_output_dir / "log",
        )

        df = pd.DataFrame(results)
        click.echo(
            f"[basis={basis_name}] AUROC (max={df.epoch_auroc.max():.4f}): {df.epoch_auroc.values[-1]:.4f}"
        )

        filename = basis_output_dir / "result.csv"

        df.to_csv(filename, index=False)

    click.echo(f"Check Results at: {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
