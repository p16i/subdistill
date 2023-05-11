from contextlib import nullcontext

import os
import pytest

from xaikd import datasets
import numpy as np

from torchvision.datasets import CIFAR10


@pytest.mark.parametrize("name", ["cifar10", "cifar100", "imagenet"])
def test_construct_dataset(name):
    dataset = datasets.construct(name)

    assert hasattr(dataset, "num_classes")
    assert hasattr(dataset, "transformation")


@pytest.mark.parametrize("name", ["cifar100-10vs99"])
def test_construct_subclasses_dataset(name):
    class1, class2 = np.array(name.split("-")[1].split("vs")).astype(int)

    dataset = datasets.construct(name)

    for train_split in [False, True]:
        for _, target in dataset.loader(train_split=train_split):
            assert np.logical_or(target == class1, target == class2).all()


@pytest.mark.parametrize(
    "name",
    ["cifar10-1999vs99", "cifar100-123vs999", "cifar100-2vs999", "cifar100-2999"],
)
def test_bad_construct_subclasses_dataset(name):
    with pytest.raises((AssertionError, ValueError)):
        _ = datasets.construct(name)


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


def test_selected_subset_samples_for_classes_adversarial():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])

    with pytest.raises(AssertionError):
        indices = datasets.selected_subset_samples_for_classes(labels, [0, 10], 2)


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
        dataset.create_dataset(train_split=train_split).targets, selected_classes
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
