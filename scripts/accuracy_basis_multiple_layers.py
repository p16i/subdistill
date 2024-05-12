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
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as F

from xaikd import models, datasets, utils, attributors, constants, prcaopt
from xaikd.utils import metrics


DEVICE = utils.get_device()
ARR_LAYER_DIMENSIONS = [
    (64, 56, 48, 40),
    (48, 40, 32, 24),
    (40, 32, 24, 16),
    (32, 24, 16, 8),
    (24, 16, 8, 4),
]


class BasisTransform:
    def rank_k_encoder(self, k: int):
        raise

    def rank_k_decoder(self, k: int):
        raise

    def get_hook_rank_k_transformation(self, k, device):

        mat_enc = self.rank_k_encoder(k)
        mat_dec = self.rank_k_decoder(k)

        # X @ U @ U.T
        mat = torch.from_numpy(mat_enc @ mat_dec)

        mat = mat.unsqueeze(2).unsqueeze(3).to(device)

        def hook_func(module, inp, out):
            return F.conv2d(out, mat)

        return hook_func


class PCA(BasisTransform):

    def __init__(self, model, layer, arr_act, arr_ctx, dataloader):

        eigvals, eigvecs = np.linalg.eigh(arr_act.T @ arr_act)

        # descending sort
        sorted_indices = np.argsort(-eigvals)
        self.eigvals = eigvals[sorted_indices]
        self.eigvecs = eigvecs[:, sorted_indices]

    def rank_k_encoder(self, k: int):
        # X @ U
        return self.eigvecs[:, :k]

    def rank_k_decoder(self, k: int):
        # Z @ U.T
        return self.eigvecs[:, :k].T


class PRCASortAbs(BasisTransform):
    def __init__(self, model, layer, arr_act, arr_ctx, dataloader):

        ccov = arr_act @ arr_ctx + arr_ctx.T @ arr_act

        eigvals, eigvecs = np.linalg.eigh(ccov)

        # descending sort
        sorted_indices = np.argsort(-np.abs(eigvals))
        self.eigvals = eigvals[sorted_indices]
        self.eigvecs = eigvecs[:, sorted_indices]

    def rank_k_encoder(self, k: int):
        # X @ U
        return self.eigvecs[:, :k]

    def rank_k_decoder(self, k: int):
        # Z @ U.T
        return self.eigvecs[:, :k].T


class PCALookAhead(BasisTransform):
    def __init__(self, model, layer, arr_act, arr_ctx, dataloader):

        self.model = model
        self.layer = layer

        _, eigvecs = np.linalg.eigh(arr_act.T @ arr_act)

        self.eigvecs = eigvecs[:, ::-1].copy()
        self.dataloader = dataloader

        self._cache = dict()

    def rank_k_encoder(self, k: int):

        if not k in self._cache:
            U = prcaopt.learn_prca_opt(
                model=self.model,
                fn=self.layer,
                location=self.layer,
                dataloader=self.dataloader,
                Uinit=self.eigvecs[:, :k],
                k=k,
                verbose=False,
            )
            self._cache[k] = U

        return self._cache[k]

    def rank_k_decoder(self, k: int):
        return self.rank_k_encoder(k).T


def get_basis_transform(
    name: str, model, layer, arr_act, arr_ctx, dataloader
) -> BasisTransform:
    if name == "pca":
        return PCA(model, layer, arr_act, arr_ctx, dataloader)
    elif name == "prcasortabs":
        return PRCASortAbs(model, layer, arr_act, arr_ctx, dataloader)
    elif name == "pcalookahead":
        return PCALookAhead(model, layer, arr_act, arr_ctx, dataloader)
    else:
        raise ValueError(f"basis={name} doesn't exist!")


@click.command()
@click.option("--dataset-name", type=str)
@click.option("--model-name", type=str)
@click.option("--bases", type=str, default="pca,prcasortabs,pcalookahead")
@click.option("--output-dir", type=str)
def main(model_name, dataset_name, output_dir, bases):
    arguments = locals()
    start_time = datetime.now()

    _, arch, _ = model_name.split("-")
    arr_layers = list(constants.ARCH_LAYER_DIMENSIONS[arch].keys())

    click.echo(f"> dataset={dataset_name}")
    click.echo(f"> model={model_name}, layers={arr_layers}")

    output_path = Path(output_dir) / dataset_name / model_name

    arr_basis_names = bases.split(",")

    dataset = datasets.construct(dataset_name)

    model = models.get_trained_model(model_name)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)

    model.to(DEVICE)

    logit_modifier = attributors.WinningClassEvidence(
        num_classes=len(dataset.selected_classes)
    )

    print(f"LogitMod: {logit_modifier}")

    dataloader_train = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )
    dataloader_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    ref_acc, _ = metrics.accuracy(
        model,
        dataloader=dataloader_val,
        num_classes=dataset.num_classes,
        device=DEVICE,
    )
    print(f"> ref_acc={ref_acc}")

    rng = np.random.default_rng(seed=1)

    for basis_name in tqdm(arr_basis_names):

        basis_output_path = output_path / basis_name
        os.makedirs(basis_output_path, exist_ok=True)

        basis_ref_acc, _ = metrics.accuracy(
            model,
            dataloader=dataloader_val,
            num_classes=dataset.num_classes,
            device=DEVICE,
        )

        np.testing.assert_allclose(
            basis_ref_acc, ref_acc, err_msg="sanity check: ref acc remains the same"
        )

        arr_statistics = []

        arr_layer_bases: list[BasisTransform] = []
        for layer in arr_layers:
            arr_act, arr_ctx = attributors.extract_activation_context(
                model=model,
                layer=layer,
                dataset=dataset,
                rng=rng,
                data_loader=dataloader_train,
                device=DEVICE,
                logit_modifier=logit_modifier,
            )

            layer_basis = get_basis_transform(
                basis_name,
                model=model,
                layer=layer,
                arr_act=arr_act,
                arr_ctx=arr_ctx,
                dataloader=datasets.build_dataloader(
                    dataset.create_subset(train_split=True),
                    shuffle=True,  # this is only used for prcalookahead
                ),
            )

            arr_layer_bases.append(layer_basis)

        for layer_dimensions in ARR_LAYER_DIMENSIONS:
            assert len(layer_dimensions) == len(arr_layers)

            arr_hooks = []
            try:
                for layer, layer_basis, k in zip(
                    arr_layers, arr_layer_bases, layer_dimensions
                ):
                    module = utils.interceptor.get_module(model=model, layer_str=layer)

                    hook = module.register_forward_hook(
                        layer_basis.get_hook_rank_k_transformation(k=k, device=DEVICE)
                    )
                    arr_hooks.append(hook)

                compressed_acc, _ = metrics.accuracy(
                    model,
                    dataloader=dataloader_val,
                    num_classes=dataset.num_classes,
                    device=DEVICE,
                )

                arr_statistics.append(
                    dict(
                        basis_name=basis_name,
                        layer_dimensions=layer_dimensions,
                        ref_acc=ref_acc,
                        acc=compressed_acc,
                    )
                )

            finally:
                for hook in arr_hooks:
                    hook.remove()

        pd.DataFrame(arr_statistics).to_csv(
            basis_output_path / f"stats.csv", index=False
        )

    click.echo(f"Check output at: {output_path}")
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
