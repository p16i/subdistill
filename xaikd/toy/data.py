import typing

import numpy as np
import numpy.typing as npt

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split

import torch

from torch.utils.data import DataLoader, TensorDataset

from dataclasses import dataclass


MEAN_GROUP = [1, 0, -1]
TEST_SPLIT_RATIO = 0.2
SAMPLES_PER_GROUP = 5000
NUM_CLASSES = 6


@dataclass
class Dataset:
    x_train: npt.NDArray
    y_train: npt.NDArray
    x_val: npt.NDArray
    y_val: npt.NDArray

    eps: float
    seed: int


def preprend_z(x: npt.NDArray, gix: int, eps: float) -> npt.NDArray:
    z = MEAN_GROUP[gix] + eps * np.random.randn(x.shape[0])

    return np.hstack([x, z.reshape((-1, 1))])


def construct_dataset(eps: float, seed: int, samples_per_group=SAMPLES_PER_GROUP):
    # These toy datasets are generated with similar parameters used in
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    np.random.seed(seed)

    xm, ym = make_moons(n_samples=samples_per_group, noise=0.3, random_state=seed)
    xm -= np.mean(xm, axis=0)
    xm = preprend_z(xm, gix=0, eps=eps)

    xc, yc = make_circles(
        n_samples=samples_per_group, noise=0.2, factor=0.5, random_state=seed
    )
    xc -= np.mean(xc, axis=0)
    yc = yc + 2  # labels: {2, 3}
    xc = preprend_z(xc, gix=1, eps=eps)

    xs, ys = make_classification(
        n_samples=samples_per_group,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=seed,
        n_clusters_per_class=1,
    )
    xs -= np.mean(xs, axis=0)
    ys = ys + 4  # labels: {4, 5}
    xs = preprend_z(xs, gix=2, eps=eps)

    X = np.vstack([xm, xc, xs])
    y = np.concatenate([ym, yc, ys])

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=seed
    )

    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    def standardize(x):
        return (x - mean) / std

    return mean, std, standardize(x_train), standardize(x_val), y_train, y_val


def build_loaders(
    dataset: Dataset, batch_size=64, num_workers=2
) -> typing.Tuple[DataLoader, DataLoader]:
    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(dataset.x_train).float(),
            torch.tensor(dataset.y_train),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_dl = DataLoader(
        TensorDataset(torch.tensor(dataset.x_val).float(), torch.tensor(dataset.y_val)),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    return train_dl, val_dl


def build_subset_loaders(
    dataset: Dataset,
    selected_classes: typing.Tuple[int, int],
    batch_size=64,
    num_workers=2,
) -> typing.Tuple[DataLoader, DataLoader]:
    arr_dl = []

    for x, y, shuffle in [
        (dataset.x_train, dataset.y_train, True),
        (dataset.x_val, dataset.y_val, False),
    ]:
        selected = np.isin(y, selected_classes)

        arr_dl.append(
            DataLoader(
                TensorDataset(
                    torch.tensor(x[selected,]).float(), torch.tensor(y[selected])
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
            )
        )

    subset_train_dl, subset_val_dl = arr_dl

    return subset_train_dl, subset_val_dl
