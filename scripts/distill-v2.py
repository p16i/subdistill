import click
import os
import pandas as pd

import numpy as np
import torch

from datetime import datetime

from pathlib import Path

from xaikd.utils import click_types
from xaikd import datasets, utils, distillators, models, attributors, bases

from tensorboard_logger import configure


@click.command()
@click.option("--dataset", default="cifar100-35vs98", type=str, required=True)
@click.option("--model", default="cifar100-resnet18-p1", required=True)
@click.option("--layer", default="layer3", type=str, required=True)
@click.option("--basis-name", type=str, default="pca", required=True)
@click.option("--compression-rate", type=float, default=0.25, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--num-samples", type=int, default=100, required=True)
@click.option("--epochs", type=int, default=50, required=True)
@click.option("--lr", type=float, default=0.001, required=True)
@click.option("--seed", type=int, default=1)
def main(
    model,
    dataset,
    basis_name,
    output_dir,
    compression_rate,
    seed,
    epochs,
    lr,
    num_samples,
    layer,
):
    np.random.seed(seed)

    arguments = locals()
    start_time = datetime.now()

    model = models.get_model(model)
    model_name = getattr(model, "__name")

    layer_slug = f"layer{layer}-n{num_samples}-comp{compression_rate}-seed{seed}"

    basis_name = f"{basis_name}--centered"

    output_dir = Path(output_dir) / dataset / model_name / layer_slug / basis_name

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    dataset: datasets.TwoClassesDataset = datasets.construct(
        dataset, num_training_samples=num_samples
    )

    logodd_mod = attributors.LogOddEvidence(dataset.selected_classes)

    # todo: make sure that all bases use the same activation and context vectors
    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
        logit_modifier=logodd_mod,
        device=device,
    )
    mean = np.mean(arr_act, axis=0)
    np.save(output_dir.parent / "act_mean", mean)

    basis = bases.get_basis(basis_name)
    basis.fit(arr_act, arr_ctx, mean=mean, device=device)
    print("basis is saved to", output_dir.parent)
    basis.save(output_dir.parent)
    print("load basis from", output_dir.parent)
    basis.load(output_dir.parent)

    distillator = distillators.Layerwise(
        teacher=model,
        dataset=dataset,
        compression_rate=compression_rate,
        device=device,
    )

    results = distillator.distill(
        epochs=epochs,
        layer=layer,
        basis=basis,
        seed=seed,
        device=device,
        lr=lr,
        log_dir=output_dir / "log",
    )

    df = pd.DataFrame(results)
    print(f"AUROC (max={df.auroc.max():.4f}): {df.auroc.values[-1]:.4f}")

    filename = output_dir / "result.csv"
    print(f"> check output at: {filename}")

    df.to_csv(filename, index=False)

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
