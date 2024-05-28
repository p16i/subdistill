import numpy as np
import pytest

from xaikd import datasets
from . import dataset_cifar100


@pytest.mark.parametrize(
    "dataset_name,expected_class_indices",
    [
        ("imagenet-butterfly", [321, 322, 323, 324, 325, 326]),
        ("imagenet-boat", [472, 554, 576, 625, 814, 914]),
        ("imagenet-car", [407, 436, 468, 511, 609, 627, 656, 661, 751, 817]),
        ("imagenet-cat", [281, 282, 283, 284, 285, 286, 287]),
        (
            "imagenet-edible_fruit",
            [
                948,
                949,
                950,
                951,
                952,
                953,
                954,
                955,
                956,
                957,
            ],
        ),
        (
            "imagenet-fungus",
            [
                991,
                993,
                994,
                995,
                996,
                997,
            ],
        ),
        (
            "imagenet-truck",
            [
                555,
                569,
                656,
                675,
                717,
                734,
                864,
                867,
            ],
        ),
    ],
)
@pytest.mark.parametrize("lvl", [0.0, 0.125, 0.25, 0.5, 1.0])
def test_dataset_accessible(dataset_name, lvl, expected_class_indices):

    if lvl > 0:
        dataset_name = "--".join([dataset_name, "spurious-copyright", f"{lvl}"])
    else:
        dataset_name = dataset_name

    dataset = datasets.construct(dataset_name)

    np.testing.assert_array_equal(dataset.selected_classes, expected_class_indices)


@pytest.mark.parametrize(
    "lvl",
    [0.125, 0.25, 0.5, 1.0],
)
@pytest.mark.parametrize("train_split", [True, False])
@pytest.mark.parametrize("dataset_slug", ["imagenet-random--spurious-copyright"])
@pytest.mark.gpu
def test_victim_propotion(dataset_slug, lvl, train_split):
    dataset_cifar100.test_dataset_with_spurious_correlation(
        dataset_slug=dataset_slug, lvl=lvl, train_split=train_split
    )
    dataset = datasets.construct("--".join([dataset_slug, str(lvl)]))

    num_classes = len(dataset.selected_classes)

    victim_class = dataset.selected_classes[0]

    ds = dataset.create_subset(train_split=train_split)

    arr_victim_indices = ds.victim_indices
    arr_targets = np.array(ds.targets)
    num_samples = arr_targets.shape[0]

    if train_split:
        np.testing.assert_allclose(
            len(arr_victim_indices),
            np.floor(lvl * (arr_targets == victim_class).sum()),
        )
        # for training set, we have victim for only for first class
        np.testing.assert_equal(arr_targets[arr_victim_indices], victim_class)
    else:
        np.testing.assert_allclose(len(arr_victim_indices), np.floor(num_samples * lvl))

        # for testing set, we have victim for all classe
        assert len(set(arr_targets[arr_victim_indices].tolist())) == num_classes
