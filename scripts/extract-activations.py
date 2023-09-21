import numpy as np
import os
import click

from datetime import datetime
from pathlib import Path


import torch

from xaikd import models, utils, attributors
from xaikd import constants
from xaikd import datasets
from xaikd import bases


@click.command()
@click.option("--model-name", type=str)
@click.option("--layer", type=str)
@click.option("--output-dir", type=str)
@click.option("--seed", default=1)
@click.option("--selected-bases", default=",".join(constants.BASIS_NAMES))
def main(model_name, layer, output_dir, seed, selected_bases):
    raise NotImplemented("need refactoring")
    arguments = locals()

    # todo: check compatability between model and layer

    device = utils.get_device()

    click.echo(f"Device: {device}")

    dataset_name, arch, variant = model_name.split("-")

    dataset = datasets.construct(dataset_name)

    start_time = datetime.now()

    torch.manual_seed(seed)

    output_dir = Path(output_dir) / model_name / layer
    os.makedirs(output_dir, exist_ok=True)

    model = models.get_trained_model(name=model_name).to(device)

    logit_modifier = attributors.OneClassEvidence(dataset)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
        logit_modifier=logit_modifier,
        device=device,
        seed=seed,
    )

    mean_act = np.mean(arr_act, axis=0)
    np.save(f"{output_dir}/act_mean", mean_act)

    for basis_name in selected_bases.split(","):
        click.echo(f"Learning {basis_name}")

        basis = bases.get_basis(basis_name)

        basis.fit(arr_act, arr_ctx, mean=mean_act, device=device)

        basis.save(output_dir)

    time_took = datetime.now() - start_time

    utils.dump_json(f"{output_dir}/meta.json", arguments)
    click.echo(f"Output: {output_dir}")
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
