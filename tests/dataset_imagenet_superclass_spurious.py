import pytest
import numpy as np
import torch

from xaikd import datasets


@pytest.mark.slow
def test():
    dataset = datasets.construct("imagenet-wading-bird--spurious-copyright--1.0")

    assert isinstance(
        dataset,
        (
            datasets.imagenet.subclasses_spurious_features.ImageNetSuperclassWithCopyrightFeatures,
            datasets.interface.WithValidationSetMixin,
        ),
    )

    trng = torch.Generator()
    trng.manual_seed(42)

    ds_train, ds_val = dataset.create_train_val_split(rng=trng, training_size=0.8)

    assert id(ds_train.dataset) != id(ds_val.dataset)

    actual_prop_spurious = np.mean(ds_train.dataset.arr_data_spurious)  # type: ignore
    np.testing.assert_allclose(actual_prop_spurious, 1 / dataset.num_classes, atol=0.05)

    np.testing.assert_allclose(
        np.sum(np.array(ds_val.dataset.arr_data_spurious) == 1), 0
    )  # type: ignore
