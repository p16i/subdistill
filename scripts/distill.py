import click
import os
from datetime import datetime

from pathlib import Path

from xaikd.utils import click_types
from xaikd import datasets, utils, distillators


@click.command()
@click.option("--model", type=click_types.Model(), required=True)
@click.option("--dataset", type=str, required=True)
@click.option(
    "--approach",
    default="grafting",
    type=click.Choice(["scratch", "hintons", "grafting", "layerwise"]),
)
@click.option("--basis-name", type=str, required=True)
@click.option("--basis-dir", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
@click.option("--compression-rate", type=float, default=0.25, required=True)
@click.option("--seed", type=int, default=1)
@click.option("--epochs", type=int, default=40, required=True)
def main(
    model,
    dataset,
    approach,
    basis_name,
    basis_dir,
    output_dir,
    compression_rate,
    seed,
    epochs,
):
    arguments = locals()
    start_time = datetime.now()

    slug = "--".join([approach, basis_name, f"comp{compression_rate}", f"seed{seed}"])

    output_dir = Path(output_dir) / dataset / slug

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    dataset: datasets.TwoClassesDataset = datasets.construct(dataset)

    distillator = distillators.Grafting(
        teacher=model,
        dataset=dataset,
        basis_dir=basis_dir,
        compression_rate=compression_rate,
        device=device,
    )

    distillator.distill(
        epochs=epochs,
        basis_name=basis_name,
        basis_dir=Path(basis_dir)
        / getattr(dataset, "__name")
        / getattr(model, "__name"),
        seed=seed,
        device=device,
    )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
