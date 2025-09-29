import pytest
import numpy as np

from xaikd import datasets


@pytest.mark.slow
def test_cifar100_and_corruption():
    dataset = datasets.construct("cifar100")
    dataset_c = datasets.construct("cifar100c")

    ds_train = dataset.create_subset(train_split=True)
    ds_train_c = dataset_c.create_subset(train_split=True)
    for attr in ["data", "targets"]:
        np.testing.assert_array_equal(
            getattr(ds_train, attr),
            getattr(ds_train_c, attr),
        )

    assert isinstance(dataset_c, datasets.cifar100c.CIFAR100C)

    ds_test_c, ds_val_c = dataset_c._create_val_test_split()

    assert isinstance(ds_test_c, datasets.cifar100c.TorchVisionCIFAR100CWithSeverity)

    assert len(ds_test_c.data) == len(ds_test_c.arr_sample_severity)
    assert len(ds_test_c.data) == 10000 * 5
    np.testing.assert_equal(
        np.bincount(ds_test_c.arr_sample_severity), [0] + [10_000] * 5
    )

    assert len(ds_val_c.data) == len(ds_val_c.arr_sample_severity)
    assert len(ds_val_c.data) == 2500 * 5
    np.testing.assert_equal(np.bincount(ds_val_c.arr_sample_severity), [0] + [2500] * 5)

    ds_test_c2 = dataset_c.create_subset(train_split=False)
    np.testing.assert_array_equal(ds_test_c.data, ds_test_c2.data)
    np.testing.assert_array_equal(ds_test_c.targets, ds_test_c2.targets)
