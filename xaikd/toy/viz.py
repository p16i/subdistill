import typing
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


import torch


from xaikd.utils import metrics
from xaikd import bases

from . import data, model as toy_model


def plot_ellipse(ax, mu, cov, n_std=1, facecolor="none", edgecolor="red", alpha=0.5):
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def dataset(artifact_dir: Path):
    X_train = np.load(artifact_dir / "x_train.npy")
    y_train = np.load(artifact_dir / "y_train.npy")

    slug = os.path.basename(artifact_dir)

    vmax = 1
    vmin = -vmax
    plt.figure(figsize=(5, 5))

    total_classes = len(set(y_train))

    plt.suptitle(
        f"{slug}: {total_classes} classes, num_training={X_train.shape[0]}", y=1.07
    )

    plt.title("Dataset")
    plt.xlim([vmin, vmax])
    plt.ylim([vmin, vmax])

    plt.axhline(0, ls="--", color="k", lw=1)
    plt.axvline(0, ls="--", color="k", lw=1)

    theta = np.linspace(0, 2 * np.pi, 150)

    xv, yv = np.meshgrid(
        np.linspace(vmin, vmax, 100),
        np.linspace(vmin, vmax, 100),
    )

    X = np.stack([xv.reshape(-1), yv.reshape(-1)]).T

    cm = plt.get_cmap("gist_rainbow")

    for i in range(total_classes):
        _x = X_train[y_train == i]
        _y = y_train[y_train == i]

        mean = np.mean(_x, axis=0)
        cov = np.cov((_x - mean).T)

        color = cm(i / 10)

        plot_ellipse(plt.gca(), mean, cov, edgecolor=color)

        plt.text(
            mean[0],
            mean[1],
            f"C{i}",
            ha="center",
            va="center",
            color=color,
            bbox=dict(facecolor="white", ec="white", ls=None, alpha=0.8),
        )

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.savefig(artifact_dir / "dataset.png", bbox_inches="tight")
    plt.close()


def subdataset_decision_boundary(
    model: torch.nn.Module,
    acc: float,
    dataset: data.Dataset,
    arr_pairs: typing.List[typing.Tuple[int, int]],
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

    ncols = data.NUM_CLASSES // 2

    plt.figure(figsize=(ncols * 4, 3 * 2))

    plt.suptitle(f"accuracy: {acc:.4f}")

    # this is seed for x3; not of the dataset
    np.random.seed(dataset.seed)

    stats_aurocs = dict()

    for gix, pix in enumerate(arr_pairs):
        plt.subplot(2, ncols, gix + 1)

        c1, c2 = dataset.arr_pairs[pix]

        _X = torch.tensor(X).float().to(device)

        _, subset_val_dl = data.build_subset_loaders(dataset, (c1, c2))

        auroc = metrics.auroc(
            model,
            subset_val_dl,
            (c1, c2),
            device=device,
        )

        auroc = np.max([auroc, 1 - auroc])

        stats_aurocs[f"p{pix}"] = auroc

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

    for kix, k in enumerate(arr_ks):
        plt.subplot(nrows, ncols, kix + 1)

        if kix == 0:
            plt.ylabel(f"{basis}")

        _X = torch.tensor(X).float()

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
