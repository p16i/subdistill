import os
import typing
import click
from tqdm import tqdm

from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt


import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from xaikd import utils, toy, bases
from xaikd.utils import metrics


@torch.no_grad()
def auroc_with_basis(
    model: nn.Module,
    module: nn.Module,
    dataloader: DataLoader,
    classes: typing.Tuple[int, int],
    basis: bases.Basis,
    device: str,
    arr_ks: typing.List[int],
) -> typing.List[float]:
    arr_aurocs = []

    for k in tqdm(arr_ks, desc=f"[basis={basis}]"):
        try:
            hook = module.register_forward_hook(
                toy.model.attach_projected_fh_with_k(k=k, basis=basis, device=device)
            )

            value = metrics.auroc(
                model,
                dataloader=dataloader,
                classes=classes,
                device=device,
            )

            arr_aurocs.append(value)

        finally:
            hook.remove()

    return arr_aurocs


def dataset_slug(eps: float, seed: int) -> str:
    return f"toy-dataset-eps{eps}-seed{seed}"


def generate_data(eps: float, seed: int, artifact_dir: Path) -> toy.data.Dataset:
    if os.path.isfile(artifact_dir / "x_train.npy"):
        print("Data is already there; we load only here.")
        artifacts = dict()
        for k in ["x_train", "y_train", "x_val", "y_val", "arr_pairs", "arr_centroids"]:
            artifacts[k] = np.load(artifact_dir / f"{k}.npy")

        return toy.data.Dataset(**artifacts, eps=eps, seed=seed)
    else:
        print(f"Generate data with eps={eps}, seed={seed}")
        for k, v in zip(
            [
                "mean",
                "std",
                "x_train",
                "x_val",
                "y_train",
                "y_val",
                "arr_pairs",
                "arr_centroids",
            ],
            toy.data.construct_dataset(eps=eps, seed=seed),
        ):
            np.save(artifact_dir / k, v)

        # visualize the dataset
        toy.viz.dataset(artifact_dir)

        return generate_data(eps, seed, artifact_dir)


def extract_activation_and_bases(
    model: nn.Module,
    module: nn.Module,
    dataset: toy.data.Dataset,
    classes: typing.Tuple[int, int],
    basis_names: typing.List[str],
    device: str,
    output_dir: Path,
) -> int:
    arr_act, arr_ctx = toy.attribution.extract_activation_context(
        model,
        module=module,
        dataset=dataset,
        selected_classes=classes,
        device=device,
    )

    print("arr_act.shape", arr_act.shape)

    mean_act = np.mean(arr_act, axis=0)
    np.save(output_dir / "act_mean", mean_act)

    for basis_name in basis_names:
        if "random" in basis_name:
            continue

        click.echo(f"Learning {basis_name}")

        basis = bases.get_basis(basis_name)

        basis.fit(arr_act, arr_ctx, mean=mean_act, device=device)

        basis.save(output_dir)

    return mean_act.shape[0]


@click.command()
@click.option("--eps", default=1.0, type=float)
@click.option("--model", default="mlp64", type=str)
@click.option("--output-dir", default="./tmp", type=str)
@click.option("--seed", default=1, type=int)
@click.option("--epochs", default=100, type=int)
@click.option(
    "--basis-names",
    type=str,
    default=",".join(
        [
            "pca",
            "prca-recon",
            "prca-abs",
            "prca",
            "random1",
            "random2",
            "random3",
        ]
    ),
)
@click.option(
    "--mode", default="centered", type=click.Choice(["centered", "uncentered"])
)
def main(model, seed, eps, output_dir, epochs, mode, basis_names):
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    output_dir = Path(output_dir) / dataset_slug(eps, seed)
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"Output dir: {output_dir}")

    # Step 1: Data preparation (generate if needed; otherwise load only)
    dataset = generate_data(eps=eps, seed=seed, artifact_dir=output_dir)

    start_time = datetime.now()

    # Step 2: Model Training
    pl.seed_everything(seed)

    model = toy.model.construct_mlp(model)

    model_output_dir = output_dir / getattr(model, "__name")
    os.makedirs(model_output_dir, exist_ok=True)

    train_loader, val_loader = toy.data.build_loaders(dataset=dataset)

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=epochs,
        deterministic=True,
        default_root_dir=model_output_dir,
    )

    trainer.fit(toy.model.ModelWrapper(model), train_loader, val_loader)

    model = model.to(device)

    with torch.no_grad():
        acc = metrics.accuracy(
            model, val_loader, num_classes=toy.data.NUM_CLASSES, device=device
        )

    arguments["accuracy"] = acc

    click.echo(f"Accuracy={acc:.4f}")

    # Step 3: Teacher Performance on Subdatasets
    # comptue auroc, viz dececision boundary?

    total_pairs = dataset.arr_pairs.shape[0]
    pair_indices = list(range(total_pairs))
    selected_pairs = pair_indices[:3] + pair_indices[-3:]

    with torch.no_grad():
        stats_auroc = toy.viz.subdataset_decision_boundary(
            model=model,
            dataset=dataset,
            arr_pairs=selected_pairs,
            acc=acc,
            device=device,
            artifact_dir=model_output_dir,
        )

    arguments["aurocs"] = stats_auroc
    utils.dump_json(model_output_dir / "meta.json", arguments)

    basis_names = list(map(lambda s: f"{s}--{mode}", basis_names.split(",")))

    for pix in selected_pairs:
        classes = dataset.arr_pairs[pix]
        cls_slug = f"subdataset--p{pix}"

        for layer in ["act1", "act2"]:
            layer_output_dir = model_output_dir / cls_slug / layer
            os.makedirs(layer_output_dir, exist_ok=True)
            module = getattr(model, layer)

            click.echo(f"[layer={layer}]: classes={classes}")

            _, subset_val_dl = toy.data.build_subset_loaders(
                dataset, selected_classes=classes
            )

            layer_dims = extract_activation_and_bases(
                model=model,
                module=module,
                dataset=dataset,
                classes=classes,
                basis_names=basis_names,
                device=device,
                output_dir=layer_output_dir,
            )

            arr_ks = list(range(0, layer_dims + 1, 1))

            for basis_name in basis_names:
                print("Basis:", basis_name)
                basis = bases.get_basis(basis_name)

                basis.load(layer_output_dir, device=device)

                arr_auroc = auroc_with_basis(
                    model=model,
                    module=module,
                    dataloader=subset_val_dl,
                    classes=classes,
                    basis=basis,
                    device=device,
                    arr_ks=arr_ks,
                )

                basis_output_dir = layer_output_dir / basis_name
                os.makedirs(basis_output_dir, exist_ok=True)

                utils.dump_json(
                    basis_output_dir / "auroc.json", dict(ks=arr_ks, auroc=arr_auroc)
                )

                with torch.no_grad():
                    toy.viz.decision_boundary_with_basis(
                        model=model,
                        dataset=dataset,
                        module=module,
                        classes=classes,
                        basis=basis,
                        arr_ks=arr_ks[:6] + [layer_dims],
                        device=device,
                        output_dir=basis_output_dir,
                    )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")
    click.echo(f"Check results at: {output_dir}")


if __name__ == "__main__":
    main()
