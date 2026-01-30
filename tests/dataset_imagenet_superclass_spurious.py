import pytest
import numpy as np
import torch

from xaikd import datasets


@pytest.mark.slow
@pytest.mark.parametrize("lvl", [0.0, 0.5, 1.0])
def test(lvl):
    dataset = datasets.construct(f"imagenet-wading-bird--mnistspurious--{lvl}")

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
    assert isinstance(
        ds_train.dataset,
        datasets.imagenet.subclasses_spurious_features.TorchVisionDatasetImageNetWithMNISTSpuriousFeatures,
    )
    total_orig_train = len(ds_train.dataset.targets)
    ds_test = dataset.create_subset(train_split=False)

    assert id(ds_train.dataset) == id(ds_val.dataset)

    assert ds_train.dataset.spurious_digit

    np.testing.assert_allclose(
        len(ds_train.dataset.arr_index_with_spurious) / total_orig_train, lvl
    )

    assert isinstance(
        ds_test,
        datasets.imagenet.subclasses_spurious_features.TorchVisionDatasetImageNetWithMNISTSpuriousFeatures,
    )
    assert ds_test.spurious_digit == False
    np.testing.assert_allclose(len(ds_test.arr_index_with_spurious) / len(ds_test), lvl)
