import typing
import numpy as np
import os
import click

from datetime import datetime
from pathlib import Path

from tqdm import tqdm


import torch
from torch.nn import functional as F
from zennit.attribution import Gradient

from xaikd import models, utils, attributors
from xaikd.constants import datasets
from xaikd import bases


def subsample_tensors(
    act: np.array, ctx: np.array, num_locations=20
) -> typing.Tuple[np.array, np.array]:
    assert len(act.shape) == 4

    bs, nc, h, w = act.shape

    total_spatial_locations = w * h
    arr_act = []
    arr_ctx = []

    for ix in range(bs):
        _a = act[ix]
        _c = ctx[ix]

        assert _a.shape == (nc, h, w)

        selected = np.random.permutation(total_spatial_locations)[:num_locations]
        flattened_act = _a.reshape((nc, -1))
        flattened_ctx = _c.reshape((nc, -1))
        selected_act = flattened_act[:, selected]
        selected_ctx = flattened_ctx[:, selected]

        arr_act.append(selected_act.T)
        arr_ctx.append(selected_ctx.T)

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    assert arr_act.shape == (bs * num_locations, nc)

    return arr_act, arr_ctx


def extract_activation_context(
    model, layer, dataset, seed=1, device="cpu"
) -> typing.Tuple[np.array, np.array]:
    dataset = datasets.get_constant(dataset)
    data_loader = dataset.loader(train_split=True)

    np.random.seed(seed)

    arr_act = []
    arr_ctx = []

    try:
        module, hook = utils.interceptor.attach_hook_intercept_output(model, layer)

        attributor: Gradient
        with attributors.make_attributor_for(
            model, dataset.input_normalization
        ) as attributor:
            for batch in tqdm(data_loader):
                x, y = batch
                x = x.to(device)

                _ = attributor.forward(
                    x,
                    lambda output: output
                    * F.one_hot(y, num_classes=dataset.num_classes).to(device),
                )

                act = utils.interceptor.get_output(module)
                rel = act.grad

                output_dimensions = act.shape[1:]

                # todo: check this with Gregoire again!
                ctx = torch.where(act.abs() > 0, rel / act, 0)

                assert torch.allclose(act * ctx, rel)

                assert ctx.shape == act.shape

                act = act.detach().cpu().numpy()
                ctx = ctx.detach().cpu().numpy()

                selected_act, selected_ctx = subsample_tensors(act, ctx)
                arr_act.append(selected_act)
                arr_ctx.append(selected_ctx)

    finally:
        hook.remove()

    print(f"{layer}: output-dims={output_dimensions}")

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    return arr_act, arr_ctx


@click.command()
@click.option("--model", type=str)
@click.option("--layer", type=str)
@click.option("--output-dir", type=str)
@click.option("--seed", default=1)
@click.option("--selected-bases", default="pca,prca,prca-abs,prca-recon")
def main(model, layer, output_dir, seed, selected_bases):
    arguments = locals()

    # todo: check compatability between model and layer

    device = utils.get_device()

    click.echo(f"Device: {device}")

    slugs = model.split("-")

    dataset = slugs[0]

    start_time = datetime.now()

    torch.manual_seed(seed)

    output_dir = Path(output_dir) / model / layer
    os.makedirs(output_dir, exist_ok=True)

    model = models.get_model(slug=model).to(device)

    # todo: how big are these array?
    arr_act, arr_ctx = extract_activation_context(
        model=model, layer=layer, dataset=dataset, device=device, seed=seed
    )

    for basis_name in selected_bases.split(","):
        click.echo(f"Learning {basis_name}")

        basis = bases.get_basis(basis_name)

        basis.fit(arr_act, arr_ctx, device=device)

        basis.save(output_dir)

    time_took = datetime.now() - start_time

    utils.dump_json(f"{output_dir}/meta.json", arguments)
    click.echo(f"Output: {output_dir}")
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
