import click
import os
from datetime import datetime

from pathlib import Path

from xaikd.utils import click_types
from xaikd import datasets, utils


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
def main(
    model, dataset, approach, basis_name, basis_dir, output_dir, compression_rate, seed
):
    arguments = locals()
    start_time = datetime.now()

    slug = "--".join([approach, basis_name, f"comp{compression_rate}", f"seed{seed}"])

    output_dir = Path(output_dir) / dataset / slug

    os.makedirs(output_dir, exist_ok=True)
    click.echo(f"Output: {output_dir}")

    device = utils.get_device()

    model = model.to(device)
    dataset: datasets.TwoClassesDataset = datasets.construct(dataset)

    # pass

    # load basis

    # inistiate: ligthing module
    # approximation approach (model, dataset)

    # train

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
