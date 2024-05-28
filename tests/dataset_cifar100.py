from contextlib import nullcontext

from collections import OrderedDict

import os
import pytest

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import CIFAR100

import pandas as pd
from copy import deepcopy

from xaikd import models, utils
from xaikd import datasets
from xaikd.utils import metrics

DF_CIFAR100_LABEL_MAPPING = pd.read_csv(
    datasets.constants.PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv"
)


@pytest.mark.parametrize("name", ["cifar100", "cifar100-people", "imagenet"])
def test_construct_dataset(name):
    dataset = datasets.construct(name)

    # todo: find a way to use @property to automatically validate these attributes
    # instead of do this manually
    assert hasattr(dataset, "num_classes")
    assert hasattr(dataset, "input_transformation")
    assert hasattr(dataset, "input_training_transformation")
    assert hasattr(dataset, "_normalizer")


@pytest.mark.parametrize(
    "super_class", DF_CIFAR100_LABEL_MAPPING["coarse_label_name"].unique().tolist()
)
def test_cifar100_superclass(super_class):
    ds: datasets.Cifar100SuperClassesDataset = datasets.construct(
        f"cifar100-{super_class}"
    )

    df = DF_CIFAR100_LABEL_MAPPING

    fine_labels = df[df.coarse_label_name == super_class].fine_label.values.tolist()

    assert tuple(sorted(ds.selected_classes)) == tuple(sorted(fine_labels))

    for ix, (_, y) in enumerate(DataLoader(ds.create_subset(train_split=False))):
        assert (y.numpy() <= ds.num_classes - 1).all()


@pytest.mark.parametrize(
    "super_class", DF_CIFAR100_LABEL_MAPPING["coarse_label_name"].unique().tolist()
)
def test_cifar100_superclass_transform_target(super_class):
    ds: datasets.Cifar100SuperClassesDataset = datasets.construct(
        f"cifar100-{super_class}"
    )

    df = DF_CIFAR100_LABEL_MAPPING

    fine_labels = sorted(
        df[df.coarse_label_name == super_class].fine_label.values.tolist()
    )

    dl_val = datasets.build_dataloader(
        ds.create_subset(train_split=False), shuffle=False
    )

    arr_ys = []
    for _, y in dl_val:
        arr_ys.extend(y.numpy().tolist())

    assert np.isin(arr_ys, range(len(fine_labels))).all()


@torch.no_grad()
@pytest.mark.slow
def test_target_transform():
    dataset_name = "cifar100-people"
    dataset = datasets.construct(dataset_name)

    model = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(3, 30, kernel_size=3)),
                ("pool1", nn.AdaptiveAvgPool2d(1)),
                ("flatten", nn.Flatten(start_dim=1)),
                ("__last_layer", nn.Linear(30, 100)),
            ]
        )
    )

    cloned_model = deepcopy(model)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)

    actual_dl_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    actual_x, actual_y = next(iter(actual_dl_val))
    assert np.isin(actual_y.numpy(), range(len(dataset.selected_classes))).all()

    actual_logits = model(actual_x)

    actual_acc = np.mean(np.argmax(actual_logits) == actual_y.numpy())

    actual_x, actual_y = next(iter(actual_dl_val))

    expected_all_logits, expected_all_y = [], []

    for x, y in DataLoader(
        CIFAR100(
            root=str(datasets.DATADIR / "cifar100"),
            train=False,
            transform=dataset.input_transformation,
        ),
        shuffle=False,
        batch_size=64,
    ):
        expected_all_logits.append(cloned_model(x).numpy())
        expected_all_y.append(y.numpy())

    expected_all_logits = np.vstack(expected_all_logits)
    expected_all_y = np.concatenate(expected_all_y)

    expected_indices = np.argwhere(
        np.isin(expected_all_y, dataset.selected_classes)
    ).reshape(-1)[: actual_logits.shape[0]]

    expected_logits = expected_all_logits[expected_indices]

    expected_acc = np.mean(
        np.argmax(expected_logits) == expected_all_y[expected_indices]
    )

    np.testing.assert_allclose(actual_acc, expected_acc)

    np.testing.assert_allclose(
        actual_logits, expected_logits[:, dataset.selected_classes], atol=1e-6
    )
