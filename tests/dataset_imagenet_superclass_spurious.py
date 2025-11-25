import pytest
import numpy as np
import torch

from xaikd import datasets


@pytest.mark.slow
def test():
    dataset = datasets.construct("imagenet-wading-bird--spurious-copyright--1.0")

    assert isinstance(
        dataset,
        datasets.imagenet.subclasses_spurious_features.ImageNetSuperclassWithCopyrightFeatures,
    )

    trng = torch.Generator()
    trng.manual_seed(42)

    ds_train, ds_val = dataset.create_train_val_split(rng=trng)

    assert id(ds_train.dataset) != id(ds_val.dataset)

    actual_prop_spurious = np.mean(ds_train.dataset.arr_data_spurious)  # type: ignore
    np.testing.assert_allclose(actual_prop_spurious, 1 / dataset.num_classes, atol=0.05)

    np.testing.assert_equal(ds_val.dataset.arr_data_spurious, 0)  # type: ignore
