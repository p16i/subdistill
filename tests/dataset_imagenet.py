import pytest

import numpy as np
import torch


from torchvision.datasets import ImageNet

from xaikd import datasets, constants


@pytest.mark.parametrize(
    "name",
    [
        "imagenet",
    ],
)
def test_construct_dataset(name):
    datasets.construct(name)
    assert True


@pytest.mark.slow()
def test_original_dataset():
    train_split = False
    dataset = datasets.construct("imagenet")

    actual_ds = dataset.create_subset(train_split=train_split)

    assert isinstance(actual_ds, ImageNet)

    expected_ds = ImageNet(
        root=str(datasets.DATADIR / "imagenet"),
        transform=dataset.input_transformation,
        split="train" if train_split else "val",
    )

    np.testing.assert_equal(len(actual_ds), len(expected_ds))

    for (actual_x, actual_y), (expected_x, expected_y) in zip(
        datasets.build_dataloader(actual_ds, shuffle=False),
        datasets.build_dataloader(expected_ds, shuffle=False),
    ):
        np.testing.assert_allclose(actual_x, expected_x)
        np.testing.assert_allclose(actual_y, expected_y)

        break


@pytest.mark.parametrize(
    "dataset_name",
    [
        "imagenet-wading-bird",
        "imagenet-retriever",
        "imagenet-working-dog",
        "imagenet-domestic-cat",
        "imagenet-bag",
        "imagenet-bottle",
        "imagenet-box",
        "imagenet-truck",
    ],
)
def test_dataset_accessible_5_classes(dataset_name):
    dataset = datasets.construct(dataset_name)

    assert dataset.num_classes == 5
