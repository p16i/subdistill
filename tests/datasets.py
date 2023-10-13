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

from xaikd import datasets, models, utils
from xaikd.utils import metrics

DF_CIFAR100_LABEL_MAPPING = pd.read_csv(
    datasets.constants.PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv"
)


@pytest.mark.parametrize("name", ["cifar10", "cifar100", "cifar100-people", "imagenet"])
def test_construct_dataset(name):
    dataset = datasets.construct(name)

    # todo: find a way to use @property to automatically validate these attributes
    # instead of do this manually
    assert hasattr(dataset, "num_classes")
    assert hasattr(dataset, "input_transformation")
    assert hasattr(dataset, "input_training_transformation")
    assert hasattr(dataset, "_normalizer")


@pytest.mark.skip("obsolete")
@pytest.mark.parametrize("name", ["cifar100-10vs99"])
def test_construct_subclasses_dataset(name):
    class1, class2 = np.array(name.split("-")[1].split("vs")).astype(int)

    dataset = datasets.construct(name)

    for train_split in [False, True]:
        for _, target in dataset.loader(train_split=train_split):
            assert np.logical_or(target == class1, target == class2).all()


@pytest.mark.skip("obsolete")
@pytest.mark.parametrize(
    "name",
    ["cifar10-1999vs99", "cifar100-123vs999", "cifar100-2vs999", "cifar100-2999"],
)
def test_bad_construct_subclasses_dataset(name):
    with pytest.raises((AssertionError, ValueError)):
        _ = datasets.construct(name)


@pytest.mark.skip("[todo]")
@pytest.mark.parametrize(
    "classes,num_samples",
    [
        ([0, 2], 3),
        ([0, 1], 2),
    ],
)
def test_selected_subset_samples_for_classes(classes, num_samples):
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    counts = np.bincount(labels)

    indices = datasets.selected_subset_samples_for_classes(labels, classes, num_samples)

    selected_labels = labels[indices]

    assert np.isin(selected_labels, classes).all()

    for cix in classes:
        expected = np.min([counts[cix], num_samples])
        assert (selected_labels == cix).sum() == expected


@pytest.mark.skip("[todo]")
def test_selected_subset_samples_for_classes_adversarial():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    with pytest.raises(AssertionError):
        indices = datasets.selected_subset_samples_for_classes(labels, [0, 10], 2)


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


@pytest.mark.parametrize(
    "name,expected",
    [
        ("cifar100", ("cifar100", None)),
        ("cifar100-55vs30", ("cifar100", "55vs30")),
    ],
)
def test_parse_dataset_name(name, expected):
    assert datasets._parse_dataset_name(name) == expected


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


@pytest.mark.slow
def test_subsample_dataset_ratio_one_no_acc_effect():
    device = utils.get_device()
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
