import typing
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt


import torch


from xaikd.utils import metrics
from xaikd import bases

from . import data, model as toy_model


def dataset(artifact_dir: Path):
    X_train = np.load(artifact_dir / "x_train.npy")
    y_train = np.load(artifact_dir / "y_train.npy")

    slug = os.path.basename(artifact_dir)

    ncols = 5

    vmin, vmax = np.min(X_train[:, :2]), np.max(X_train[:, :2])
    vmin -= 0.5
    vmax += 0.5
    plt.figure(figsize=(ncols * 3.5, 3))

    total_classes = len(set(y_train))

    plt.suptitle(
        f"{slug}: {total_classes} classes, num_training={X_train.shape[0]}", y=1.07
    )

    plt.subplot(1, ncols, 1)
    plt.title("Dataset")
    plt.xlim([vmin, vmax])
    plt.ylim([vmin, vmax])

    markers = ["s", "o", "x"]

    plt.axhline(0, ls="--", color="k", lw=1)
    plt.axvline(0, ls="--", color="k", lw=1)

    for group in range(0, 6, 2):
        pos = y_train == group
        neg = y_train == (group + 1)

        plt.scatter(
            X_train[pos, 0],
            X_train[pos, 1],
            label=f"Group {group}: Pos",
            marker=markers[group // 2],
            alpha=0.1,
        )
        plt.scatter(
            X_train[neg, 0],
            X_train[neg, 1],
            label=f"Group {group}: Neg",
            marker=markers[group // 2],
            alpha=0.1,
        )

    plt.xlabel("$x_1$ (standardized)")
    plt.ylabel("$x_2$ (standardized)")

    for group in range(0, 6, 2):
        plt.subplot(1, ncols, group // 2 + 2)
        plt.axhline(0, ls="--", color="k", lw=1)
        plt.axvline(0, ls="--", color="k", lw=1)

        plt.title(f"Subdataset {group // 2}")
        plt.xlim([vmin, vmax])
        plt.ylim([vmin, vmax])

        c1, c2 = group, group + 1
        pos = y_train == c1
        neg = y_train == c2

        plt.scatter(
            X_train[pos, 0],
            X_train[pos, 1],
            label=f"Class {c1}",
            marker=markers[group // 2],
            alpha=0.1,
        )
        plt.scatter(
            X_train[neg, 0],
            X_train[neg, 1],
            label=f"Class {c2}",
            marker=markers[group // 2],
            alpha=0.1,
        )
        plt.legend()
        plt.yticks([])

    plt.subplot(1, ncols, 5)

    for group in range(0, 6, 2):
        _x = X_train[np.isin(y_train, [group, group + 1]), :]
        plt.scatter([group // 2] * _x.shape[0], _x[:, 2], alpha=0.5, marker=".")
        plt.scatter([group // 2], [_x[:, 2].mean()], marker="x", color="k")

    plt.ylabel("$x_3$ (standardized)")

    ticks = list(range(3))
    plt.xticks(ticks, list(map(lambda t: f"Dataset {t}", ticks)))
    plt.yticks([-1, 0, 1])
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.25, hspace=0.0
    )

    plt.savefig(artifact_dir / "dataset.png", bbox_inches="tight")
    plt.close()


def subdataset_decision_boundary(
    model: torch.nn.Module,
    acc: float,
    dataset: data.Dataset,
    device: str,
    artifact_dir: Path,
):
    vmin, vmax = np.min(dataset.x_val[:, :2]), np.max(dataset.x_val[:, :2])
    vmin -= 0.5
    vmax += 0.5

    arr_logits = []
    arr_targets = []

    _, val_loader = data.build_loaders(dataset)

    for x, y in val_loader:
        logits = model(x.to(device)).cpu().numpy()
        arr_logits.append(logits)
        arr_targets.extend(y.numpy().tolist())

    arr_logits = np.vstack(arr_logits)
    arr_targets = np.array(arr_targets)

    np.testing.assert_allclose(arr_targets, dataset.y_val)

    xv, yv = np.meshgrid(
        np.linspace(vmin, vmax, 100),
        np.linspace(vmin, vmax, 100),
    )

    X = np.stack([xv.reshape(-1), yv.reshape(-1)]).T

    ncols = 3

    plt.figure(figsize=(ncols * 4, 3 * 2))

    plt.suptitle(f"accuracy: {acc:.4f}")

    # this is seed for x3; not of the dataset
    np.random.seed(dataset.seed)

    stats_aurocs = dict()

    for group in range(0, 6, 2):
        gix = group // 2
        plt.subplot(2, ncols, gix + 1)

        _X = (
            torch.tensor(data.preprend_z(X, gix=gix, eps=dataset.eps))
            .float()
            .to(device)
        )
        c1, c2 = group, group + 1

        _, subset_val_dl = data.build_subset_loaders(dataset, (c1, c2))

        auroc = metrics.auroc(
            model,
            subset_val_dl,
            (c1, c2),
            device=device,
        )

        auroc = np.max([auroc, 1 - auroc])

        stats_aurocs["{c1}vs{c2}"] = auroc

        logits = model(_X).cpu()

        logodd = logits[:, c1] - logits[:, c2]

        plt.title(f"Subdataset {gix}: AUROC={auroc:.4f}")

        plt.contourf(xv, yv, logodd.reshape(yv.shape), levels=10, cmap="RdBu", alpha=1)

        for cix, c in enumerate([c1, c2]):
            selected = np.argwhere(dataset.y_val == c).reshape(-1)
            selected = np.random.permutation(selected)[:200]
            plt.scatter(
                dataset.x_val[selected, 0],
                dataset.x_val[selected, 1],
                marker=".",
                ec="k",
                alpha=0.5,
            )

        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")

        plt.subplot(2, ncols, gix + 1 + ncols)

        for cix, c in enumerate([c1, c2]):
            selected = np.argwhere(dataset.y_val == c).reshape(-1)
            logits = arr_logits[selected,]
            plt.scatter(
                logits[:, c1] - logits[:, c2],
                dataset.x_val[selected, 2],
                marker=".",
                alpha=0.3,
            )
        if gix == 0:
            plt.ylabel("$x_3$")
        plt.xlabel(f"$f_{c1} - f_{c2}$")
    plt.savefig(artifact_dir / "decision_boundary.png")
    plt.close()

    return stats_aurocs


def decision_boundary_with_basis(
    model: torch.nn.Module,
    dataset: data.Dataset,
    module: torch.nn.Module,
    classes: typing.Tuple[int, int],
    basis: bases.Basis,
    arr_ks: typing.List[int],
    device: str,
    output_dir: str,
):
    vmin, vmax = np.min(dataset.x_val[:, :2]), np.max(dataset.x_val[:, :2])
    vmin -= 0.5
    vmax += 0.5

    xv, yv = np.meshgrid(
        np.linspace(vmin, vmax, 100),
        np.linspace(vmin, vmax, 100),
    )

    X = np.stack([xv.reshape(-1), yv.reshape(-1)]).T

    ncols = len(arr_ks)
    nrows = 1

    plt.figure(figsize=(ncols * 4, 3 * nrows))

    np.random.seed(dataset.seed)

    plt.suptitle(f"eps={dataset.eps}; seed={dataset.seed}", y=1.02)

    c1, c2 = classes
    gix = c1 // 2

    for kix, k in enumerate(arr_ks):
        plt.subplot(nrows, ncols, kix + 1)

        if kix == 0:
            plt.ylabel(f"{basis}")

        _X = torch.tensor(data.preprend_z(X, gix, dataset.eps)).float()

        try:
            hook = module.register_forward_hook(
                toy_model.attach_projected_fh_with_k(
                    k=k,
                    basis=basis,
                    device=device,
                )
            )
            logits = model(_X.to(device)).cpu()

            _, subset_val_dl = data.build_subset_loaders(
                dataset, selected_classes=classes
            )

            auroc = metrics.auroc(model, subset_val_dl, classes=classes, device=device)

            auroc = np.max([auroc, 1 - auroc])

        finally:
            hook.remove()

        plt.title(f"k={k}(AUROC={auroc:.4f})")

        logodd = logits[:, c1] - logits[:, c2]

        plt.contourf(xv, yv, logodd.reshape(yv.shape), levels=10, cmap="RdBu", alpha=1)

        for c in [c1, c2]:
            selected = np.argwhere(dataset.y_val == c).reshape(-1)
            selected = np.random.permutation(selected)[:500]
            plt.scatter(
                dataset.x_val[selected, 0],
                dataset.x_val[selected, 1],
                marker=".",
                ec="k",
                alpha=0.4,
            )

    plt.savefig(output_dir / "decision_boundary.png")
    plt.close()
