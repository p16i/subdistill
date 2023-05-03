import pytest

from xaikd import datasets
import numpy as np


def test_construct_dataset():
    const_dataset = datasets.construct("cifar10")

    assert hasattr(const_dataset, "num_classes")
    assert hasattr(const_dataset, "transformation")


def test_construct_subclasses_dataset():
    class1 = 10
    class2 = 99
    name = f"cifar100-{class1}vs{class2}"

    dataset = datasets.construct(name)

    for train_split in [False, True]:
        for _, target in dataset.loader(train_split=train_split):
            assert np.logical_or(target == class1, target == class2).all()


@pytest.mark.parametrize(
    "name", ["cifar10-1999vs99", "cifar100-123vs999", "cifar100-2vs999", "cifar100-2999"]
)
def test_bad_construct_subclasses_dataset(name):
    with pytest.raises(AssertionError):
        _ = datasets.construct(name)
