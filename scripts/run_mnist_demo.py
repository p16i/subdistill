import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
import click
from datetime import datetime
from pathlib import Path

from pytorch_lightning.loggers import TensorBoardLogger

import xaikd.mnist_demo as mnist_demo

import torch

import pytorch_lightning as pl

from datetime import datetime

from xaikd import utils
from xaikd.utils import metrics
from xaikd.bases import get_basis, Basis
from torch.utils.data import DataLoader


import itertools


def train_approximator(
    teacher_model: mnist_demo.CNN,
    basis: Basis,
    k: int,
    lambda_mse: float,
    lambda_crossent: float,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model_path: Path,
    epochs=50,
    device="cpu",
    verbose=False,
) -> mnist_demo.approximator.Approximator:
    pl.seed_everything(mnist_demo.SEED)

    approximator = mnist_demo.approximator.Approximator(
        k, kernel_size=teacher_model.kernel_size
    )

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=epochs,
        enable_checkpointing=False,
        deterministic=True,
        logger=TensorBoardLogger(model_path),
    )

    trainer.fit(
        mnist_demo.approximator.ApproximatorModelWrapper(
            approximator,
            teacher_model,
            basis=basis,
            k=k,
            lambda_mse=lambda_mse,
            lambda_xent=lambda_crossent,
            verbose=verbose,
            device=device,
        ),
        train_dataloader,
        val_dataloader,
    )

    return approximator


@click.command()
@click.option("--model-name", default="mnist-k14-h128", type=str)
@click.option("--epochs", default=100, type=int)
@click.option("--output-dir", default="./tmp", type=str)
@click.option("--samples-per-class", default=7000, type=int)
def main(model_name, epochs, output_dir, samples_per_class):
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    output_dir = Path(output_dir) / f"{model_name}-n{samples_per_class}"
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"Result save to {output_dir}")

    teacher_model = mnist_demo.get_model(model_name)
    _, val_loader = mnist_demo.get_loaders()

    teacher_model.to(device=device)

    train_subclass_ds, val_subclass_ds = mnist_demo.build_subclasses_loader(
        mnist_demo.CONSIDERED_CLASSES, samples_per_class=samples_per_class
    )
    val_subset_loader = DataLoader(
        val_subclass_ds,
        num_workers=mnist_demo.NUM_WORKERS,
        batch_size=mnist_demo.BATCH_SIZE,
    )

    teacher_acc = metrics.accuracy(
        teacher_model, val_loader, mnist_demo.NUMBER_CLASSES, device=device
    )

    teacher_auroc, _ = metrics.auroc(
        teacher_model,
        val_subset_loader,
        classes=mnist_demo.CONSIDERED_CLASSES,
        device=device,
        should_convert_auroc=True,
    )

    click.echo(f"Teacher Model: {teacher_model} (acc={teacher_acc:.4f})")
    click.echo(f"> AUROC({mnist_demo.CONSIDERED_CLASSES})={teacher_auroc:.4f})")

    # Step 1: Estimate Basis

    arr_act, arr_ctx = mnist_demo.extract_activaiton_and_context(
        teacher_model,
        layer=mnist_demo.CONSIDERED_LAYER,
        train_subset=train_subclass_ds,
        device=device,
    )

    mean = arr_act.mean(axis=0)
    os.makedirs(output_dir / mnist_demo.CONSIDERED_LAYER, exist_ok=True)
    np.save(output_dir / mnist_demo.CONSIDERED_LAYER / "act_mean.npy", mean)

    stats_basis_accuracy = []

    for basis_name in ["identity"] + mnist_demo.BASIS_CONSIDERED + ["rel", "rel-abs"]:
        basis = get_basis(f"{basis_name}--centered")

        basis.fit(arr_act, arr_ctx, mean=mean, device=device)

        basis.save(output_dir / mnist_demo.CONSIDERED_LAYER)
        basis.load(output_dir / mnist_demo.CONSIDERED_LAYER, device=device)

        aurocs = metrics.auroc_with_basis(
            model=teacher_model,
            module=getattr(teacher_model, mnist_demo.CONSIDERED_LAYER),
            dataloader=val_subset_loader,
            classes=tuple(mnist_demo.CONSIDERED_CLASSES),
            basis=basis,
            device=device,
            arr_ks=mnist_demo.ARRAY_KS,
            should_convert_auroc=True,
        )
        for i in range(mnist_demo.ARRAY_KS.shape[0]):
            k = mnist_demo.ARRAY_KS[i]
            auroc, _ = aurocs[i]

            stats_basis_accuracy.append(dict(basis=basis_name, k=k, auroc=auroc))

    df_stats_basis_accuracy = pd.DataFrame(stats_basis_accuracy)
    df_stats_basis_accuracy.to_csv(output_dir / "basis_auroc.csv")

    train_subset_loader = DataLoader(
        train_subclass_ds,
        num_workers=mnist_demo.NUM_WORKERS,
        batch_size=mnist_demo.BATCH_SIZE,
        shuffle=True,
    )

    for basis_name in mnist_demo.BASIS_CONSIDERED:
        basis = get_basis(f"{basis_name}--centered")

        basis.load(output_dir / mnist_demo.CONSIDERED_LAYER, device=device)
        stats_approximator = []

        for k in np.arange(1, 5 + 1):
            for lambda_mse, lambda_crossent in itertools.product(
                mnist_demo.ARRAY_LAMBDA, mnist_demo.ARRAY_LAMBDA
            ):
                model_path = (
                    output_dir
                    / mnist_demo.CONSIDERED_LAYER
                    / "models"
                    / basis_name
                    / f"k{k}-lm{lambda_mse}-lx{lambda_crossent}"
                )
                print(f"Working with {model_path}")
                os.makedirs(model_path, exist_ok=True)
                approx = train_approximator(
                    teacher_model,
                    basis=basis,
                    model_path=model_path,
                    k=k,
                    lambda_mse=lambda_mse,
                    lambda_crossent=lambda_crossent,
                    train_dataloader=train_subset_loader,
                    val_dataloader=val_subset_loader,
                    epochs=epochs,
                    device=device,
                )

                np.save(
                    model_path / "weight", approx.conv1.weight.detach().cpu().numpy()
                )
                np.save(model_path / "bias", approx.conv1.bias.detach().cpu().numpy())

                model_with_approx = mnist_demo.approximator.CombinedModule(
                    approximator=approx,
                    teacher=teacher_model,
                    basis=basis,
                    device=device,
                )

                row = dict(
                    name=basis_name,
                    k=k,
                    lambda_mse=lambda_mse,
                    lambda_crossent=lambda_crossent,
                )

                for set_name, data_loader in [
                    ("train", train_subset_loader),
                    ("val", val_subset_loader),
                ]:
                    auroc, _ = metrics.auroc(
                        model_with_approx,
                        data_loader,
                        classes=tuple(mnist_demo.CONSIDERED_CLASSES),
                        device=device,
                        should_convert_auroc=True,
                    )

                    row[f"auroc_{set_name}"] = auroc

                stats_approximator.append(row)

        df_stats_approx = pd.DataFrame(stats_approximator)
        df_stats_approx.to_csv(output_dir / f"{basis_name}_stats_approx.csv")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")
    click.echo(f"Check results at {output_dir}")


if __name__ == "__main__":
    main()
