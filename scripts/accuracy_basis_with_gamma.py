import re
import typing
import os
import click
from pathlib import Path

from datetime import datetime
from matplotlib import pyplot as plt
from tqdm import tqdm

import numpy as np
import numpy.typing as npt
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch.nn import functional as F

from xaikd import models, datasets, utils, attributors
from xaikd.utils import metrics

from zennit.torchvision import ResNetCanonizer
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient

DEVICE = utils.get_device()


def make_attributor_for(
    model: nn.Module,
    input_statistics,
    gamma: float,
) -> Gradient:
    input_transform = transforms.Normalize(*input_statistics)

    low, high = input_transform(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))

    if isinstance(model, models.resnet.resnet.ResNet):
        canonizers = [ResNetCanonizer()]
    else:
        canonizers = []

    print(f"Instantiating EpsilonGammaBox(gamma={gamma})")

    composite = EpsilonGammaBox(low=low, high=high, canonizers=canonizers, gamma=gamma)

    return Gradient(model=model, composite=composite)


def extract_activation_context(
    model: nn.Module,
    layer: str,
    dataset: datasets.DatasetConfiguration,
    gamma: float,
    device: str,
    number_of_selected_spatial_locations=20,
    verbose=False,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    logit_modifier = attributors.WinningClassEvidence(
        num_classes=len(dataset.selected_classes)
    )
    rng = np.random.default_rng(seed=1)

    data_loader = datasets.build_dataloader(
        # training set
        dataset.create_subset(train_split=True),
        shuffle=False,
    )

    arr_act = []
    arr_ctx = []

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=True
        )

        with make_attributor_for(
            model, dataset.input_statistics, gamma=gamma
        ) as attributor:
            for batch in tqdm(data_loader):
                x, y = batch
                x = x.to(device)

                _ = attributor.forward(x, lambda logits: logit_modifier(logits, y))

                act = utils.interceptor.get_output(module)
                rel = act.grad

                output_dimensions = act.shape[1:]

                # todo: check this with Gregoire again!
                ctx = torch.where(act.abs() > 0, rel / act, 0)

                assert torch.allclose(act * ctx, rel)

                assert ctx.shape == act.shape

                act = act.detach().cpu().numpy()
                ctx = ctx.detach().cpu().numpy()

                selected_act, selected_ctx = utils.subsample_tensors(
                    act,
                    ctx,
                    num_locations=number_of_selected_spatial_locations,
                    rng=rng,
                )
                arr_act.append(selected_act)
                arr_ctx.append(selected_ctx)

    finally:
        hook.remove()
    if verbose:
        print(f"{layer}: output-dims={output_dimensions}")

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    return arr_act, arr_ctx


def fh_low_rank(U: npt.NDArray) -> typing.Tuple[torch.Tensor, typing.Callable]:
    # U.shape = (d, k)
    d, k = U.shape

    assert k <= d

    UUT = U @ U.T
    assert UUT.shape == (d, d)

    UUT = torch.from_numpy(UUT).float().unsqueeze(2).unsqueeze(3).to(DEVICE)

    def do(module, inp, out):
        (inp,) = inp

        out2 = F.conv2d(out, UUT)

        if k == d and False:
            np.testing.assert_allclose(
                out.detach().cpu().numpy(), out2.detach().cpu().numpy(), atol=1e-3
            )

        return out2

    return UUT, do


def estimate_basis(basis_name, arr_act, arr_ctx) -> npt.NDArray:
    _, d = arr_act.shape

    if basis_name == "pca":
        _, eigvecs = np.linalg.eigh(arr_act.T @ arr_act)
        U = eigvecs[:, ::-1].copy()
    elif basis_name == "prcasortabs":
        eigvals, eigvecs = np.linalg.eigh(arr_act.T @ arr_ctx + arr_ctx.T @ arr_act)

        indices = np.argsort(-np.abs(eigvals))

        U = eigvecs[:, indices].copy()
    elif re.match(r"pca([\.\d]+)prcasortabs", basis_name):
        ratio = float(re.match(r"pca([\.\d]+)prcasortabs", basis_name).group(1))
        K = int(np.floor(d * ratio))
        print(f"Constructing `{basis_name}` (with K={K})")
        Upca = estimate_basis("pca", arr_act, arr_ctx)
        Upca_K = Upca[:, :K]
        Uprca = estimate_basis("prcasortabs", arr_act @ Upca_K, arr_ctx @ Upca_K)
        U = Upca_K @ Uprca
    else:
        raise ValueError(f"no basis={basis_name}")

    return U


def compute_accuracy_of_basis_at_k(
    model: nn.Module,
    dataset: datasets.DatasetConfiguration,
    layer: str,
    U: npt.NDArray,
    arr_ks: typing.List[int],
) -> pd.DataFrame:
    ds_val = dataset.create_subset(train_split=False)
    dl = datasets.build_dataloader(ds_val, shuffle=False)

    module = utils.interceptor.get_module(model, layer)

    rows = []

    for k in tqdm(arr_ks):
        Uk = U[:, :k]
        try:
            UUT, hook_func = fh_low_rank(Uk)
            hook = module.register_forward_hook(hook_func)
            acc, _ = metrics.accuracy(
                model,
                dataloader=dl,
                num_classes=dataset.num_classes,
                device=DEVICE,
            )

            rows.append(dict(layer=layer, k=k, acc=acc))

        finally:
            hook.remove()
            del UUT

    df = pd.DataFrame(rows)

    return df


@click.command()
@click.option("--dataset-name", type=str)
@click.option("--model-name", type=str)
@click.option("--layers", type=str)
@click.option("--bases", type=str)
@click.option("--gamma", type=float, default=0.25)
@click.option("--output-dir", type=str)
def main(model_name, dataset_name, layers, gamma, output_dir, bases):
    arguments = locals()
    start_time = datetime.now()

    arr_layers = layers.split(",")

    click.echo(f">> model={model_name};  dataset={dataset_name}")

    output_path = Path(output_dir) / dataset_name / model_name

    arr_basis_names = bases.split(",")

    dataset = datasets.construct(dataset_name)

    model = models.get_trained_model(model_name)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)

    model.to(DEVICE)

    for layer in tqdm(arr_layers):
        layer_output_path = output_path / layer
        os.makedirs(layer_output_path, exist_ok=True)

        arr_act, arr_ctx = extract_activation_context(
            model=model, layer=layer, dataset=dataset, gamma=gamma, device=DEVICE
        )

        _, dims = arr_act.shape

        arr_ks = (
            np.unique(np.array(utils.logspace(dims) + list(range(1, 31 + 1))))
            .astype(int)
            .tolist()
        )

        for basis_name in arr_basis_names:
            U = estimate_basis(
                basis_name=basis_name,
                arr_act=arr_act,
                arr_ctx=arr_ctx,
            )
            df = compute_accuracy_of_basis_at_k(
                model=model, dataset=dataset, layer=layer, U=U, arr_ks=arr_ks
            )

            filepath = layer_output_path / f"{basis_name}--gamma{gamma}.csv"

            df.to_csv(filepath, index=False)

    click.echo(f"Check output at: {output_path}")
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
