import os
import pytest

from xaikd import datasets
import numpy as np


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
    with pytest.raises(AssertionError):
        _ = datasets.construct(name)
