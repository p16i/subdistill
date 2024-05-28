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


@pytest.mark.skip("[todo]")
@pytest.mark.parametrize("train_split", [True, False])
def test_subset_with_without_num_samples(train_split):
    selected_classes = [0, 2]
    num_train_samples = 7

    dataset = datasets.construct("cifar100")

    subset_full = datasets.TwoClassesDataset(dataset, selected_classes=selected_classes)

    subset_small = datasets.TwoClassesDataset(
        dataset, selected_classes=selected_classes, num_train_samples=7
    )

    def count_in_loader(dl: datasets.DataLoader):
        count = 0

        for _, y in dl:
            count += len(y)

        return count

    expected = np.isin(
        dataset.create_subset(train_split=train_split).targets, selected_classes
    ).sum()

    with nullcontext():
        assert count_in_loader(subset_full.loader(train_split=train_split)) == expected

    with nullcontext():
        if train_split:
            assert count_in_loader(
                subset_small.loader(train_split=train_split)
            ) == num_train_samples * len(selected_classes)
        else:
            assert (
                count_in_loader(subset_small.loader(train_split=train_split))
                == expected
            )


def test_subsample_dataset():
    ratio = 0.1
    dataset_name = "cifar100-people"
    dataset = datasets.construct(dataset_name)

    expected = 50 * 5

    actual = 0
    for x, _ in DataLoader(
        datasets.subsample_dataset(
            dataset=dataset.create_subset(train_split=True), ratio=ratio, seed=1
        ),
        batch_size=100,
    ):
        actual += x.shape[0]

    assert actual == expected


def test_same_seed_same_subsampled_datasets():
    dataset_name = "cifar100-people"
    dataset = datasets.construct(dataset_name)

    ds = dataset.create_subset(train_split=True)
    ratio = 0.1

    sub1 = datasets.subsample_dataset(ds, ratio=ratio, seed=1)
    sub2 = datasets.subsample_dataset(ds, ratio=ratio, seed=1)
    sub3 = datasets.subsample_dataset(ds, ratio=ratio, seed=2)

    np.testing.assert_equal(sub1.indices, sub2.indices)
    assert np.not_equal(sub1.indices, sub3.indices).any()


@pytest.mark.slow
def test_subsample_dataset_ratio_one_no_acc_effect():
    device = utils.get_device()
    # todo: also tested imagenet
    dataset_name = "cifar100-people"
    dataset = datasets.construct(dataset_name)
    num_classes = len(dataset.selected_classes)

    model = models.get_trained_model("cifar100-resnet18-v1")
    model.to(device)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)

    ds_train = dataset.create_subset(train_split=True)

    ds_train_2 = datasets.subsample_dataset(ds_train, ratio=1.0, seed=1)

    expected_acc = metrics.accuracy(
        model,
        datasets.build_dataloader(ds_train, shuffle=False),
        num_classes=num_classes,
        device=device,
    )

    actual_acc = metrics.accuracy(
        model,
        datasets.build_dataloader(ds_train_2, shuffle=False),
        num_classes=num_classes,
        device=device,
    )

    np.testing.assert_allclose(actual_acc, expected_acc, atol=1e-7)
