import typing

import numpy as np
import numpy.typing as npt

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn import cluster

import torch

from torch.utils.data import DataLoader, TensorDataset

from dataclasses import dataclass


MEAN_GROUP = [1, 0, -1]
TEST_SPLIT_RATIO = 0.2
SAMPLES_PER_BLOB = 1000
NUM_CLASSES = 10


@dataclass
class Dataset:
    x_train: npt.NDArray
    y_train: npt.NDArray
    x_val: npt.NDArray
    y_val: npt.NDArray

    arr_pairs: npt.NDArray
    arr_centroids: npt.NDArray

    eps: float
    seed: int


def preprend_z(x: npt.NDArray, gix: int, eps: float) -> npt.NDArray:
    z = MEAN_GROUP[gix] + eps * np.random.randn(x.shape[0])

    return np.hstack([x, z.reshape((-1, 1))])


def construct_dataset(
    eps: float, seed: int, samples_per_blob=SAMPLES_PER_BLOB, nblobs=NUM_CLASSES
):
    # These toy datasets are generated with similar parameters used in
    # https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    np.random.seed(seed)

    scale = 10

    points = scale * np.random.rand(100 * nblobs, 2) - scale / 2

    kmean = cluster.KMeans(nblobs, random_state=seed)

    kmean.fit(points)

    arr_centroids = kmean.cluster_centers_
    # (quasi) canonicalize cluster id
    _dix = np.argsort(np.linalg.norm(arr_centroids - [-scale / 2, scale / 2], axis=1))
    arr_centroids = arr_centroids[_dix, :]

    arr_x = []

    arr_targets = []

    for bix in range(nblobs):
        _mu = arr_centroids[bix, :]

        theta = np.random.uniform(low=0, high=2 * np.pi)

        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        arr_x.append(
            _mu
            + (np.array([eps, eps / 2]) * np.random.randn(samples_per_blob, 2)) @ rot
        )

        arr_targets.append([bix] * samples_per_blob)

    arr_pairs = []
    arr_dist = []

    for c1 in range(nblobs - 1):
        for c2 in range(c1 + 1, nblobs):
            dist = np.linalg.norm(arr_centroids[c1, :] - arr_centroids[c2, :])
            arr_dist.append(dist)
            arr_pairs.append((c1, c2))

    arr_pairs = np.array(arr_pairs)
    arr_dist = np.array(arr_dist)
    indices = np.argsort(arr_dist)

    arr_pairs = arr_pairs[indices]
    arr_dist = arr_dist[indices]

    X = np.vstack(arr_x)
    y = np.concatenate(arr_targets)

    X = X / (scale / 2)

    assert X.shape == (samples_per_blob * nblobs, 2)
    assert y.shape == (samples_per_blob * nblobs,)

    x_train, x_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SPLIT_RATIO, random_state=seed
    )

    mean = np.mean(x_train, axis=0)
    std = np.ones_like(mean)

    def normalize(x):
        return (x - mean) / std

    return (
        mean,
        std,
        normalize(x_train),
        normalize(x_val),
        y_train,
        y_val,
        arr_pairs,
        arr_centroids,
    )


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
