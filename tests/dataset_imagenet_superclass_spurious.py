import pytest
import numpy as np
import torch

from xaikd import datasets


@pytest.mark.slow
def test():
    dataset = datasets.construct("imagenet-wading-bird--mnistspurious--1.0")

    assert isinstance(
        dataset,
        (
            datasets.imagenet.subclasses_spurious_features.ImageNetSuperclassWithMNISTSpuriousFeatures,
            datasets.interface.WithValidationSetMixin,
        ),
    )

    trng = torch.Generator()
    trng.manual_seed(42)

    training_size = 0.8
    ds_train, ds_val = dataset.create_train_val_split(
        rng=trng, training_size=training_size
    )
    ds_test = dataset.create_subset(train_split=False)

    assert id(ds_train.dataset) != id(ds_val.dataset)

    np.testing.assert_allclose(
        len(ds_train.dataset.arr_index_with_spurious)  # type: ignore
        * training_size,  # the index is for the original training size, and we factor the training size split
        len(ds_train),
        atol=5,
    )

    np.testing.assert_allclose(len(ds_test.arr_index_with_spurious), len(ds_test))
