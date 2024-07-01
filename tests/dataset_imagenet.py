import numpy as np
import pytest

from xaikd import datasets, constants
from tests import dataset_cifar100
from torchvision.datasets import ImageNet


@pytest.mark.parametrize(
    "dataset_name,expected_class_indices",
    [
        ("imagenet-butterfly", [321, 322, 323, 324, 325, 326]),
        ("imagenet-boat", [472, 554, 576, 625, 814, 914]),
        ("imagenet-car", [407, 436, 468, 511, 609, 627, 656, 661, 751, 817]),
        ("imagenet-cat", [281, 282, 283, 284, 285, 286, 287]),
        (
            "imagenet-valsplit-cat--spurious-copyrightwatermarkjpeg--0.0",
            [281, 282, 283, 284, 285, 286, 287],
        ),
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

    arr_datasets = []
    if lvl > 0:
        for cix in expected_class_indices:
            arr_datasets.append(
                "--".join([dataset_name, f"spurious-watermarkC{cix}", f"{lvl}"])
            )
    else:
        arr_datasets = [dataset_name]

    for dataset in arr_datasets:
        dataset = datasets.construct(dataset_name)

        np.testing.assert_array_equal(dataset.selected_classes, expected_class_indices)


@pytest.mark.parametrize(
    "lvl",
    [1.0],
)
@pytest.mark.parametrize("train_split", [True, False])
@pytest.mark.parametrize("dataset_slug", ["imagenet-random--spurious-watermark"])
@pytest.mark.parametrize(
    "cix,victim_class",
    [(0, 100)],
)
@pytest.mark.gpu
def test_victim_propotion(dataset_slug, cix, victim_class, lvl, train_split):
    dataset_name = f"{dataset_slug}C{cix}"
    dataset_cifar100.test_dataset_with_spurious_correlation(
        dataset_slug=dataset_name,
        lvl=lvl,
        train_split=train_split,
        victim_class=victim_class,
    )
    dataset = datasets.construct("--".join([dataset_name, str(lvl)]))

    num_classes = len(dataset.selected_classes)

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


@pytest.mark.parametrize("lvl", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("train_split", [True, False])
# @pytest.mark.parametrize(
#     "variant",
#     [
#         "spurious-copyright",
#         "spurious-watermark",
#         "spurious-jpeg",
#     ],
# )
@pytest.mark.parametrize(
    "cix, victim_class",
    list(enumerate(datasets.imagenet.IMAGENET_SUPERCLASS_MAPPING["butterfly"])),
)
@pytest.mark.slow
@pytest.mark.skip("skip for nowt c")
def test_valsplit_dataset_with_spurious_correlation(
    lvl, train_split, cix, victim_class
):
    variant = f"spurious-copyrightwatermarkjpeg"
    total_train_samples = len(
        datasets.construct("imagenet-butterfly").create_subset(train_split=True).targets
    )
    # testing the size of the split
    dataset = datasets.construct(f"imagenet-valsplit-butterfly--{variant}--{lvl}")
    # remark: here, we get subset of the official training set
    ds = dataset.create_subset(train_split=train_split)

    assert ds.split == "train"

    assert isinstance(ds, ImageNet)

    num_samples = len(ds.targets)

    np.testing.assert_allclose(
        num_samples,
        total_train_samples
        * (
            constants.TRAINING_VAL_SPLIT_RATIO
            if train_split
            else 1 - constants.TRAINING_VAL_SPLIT_RATIO
        ),
    )

    victim_class = victim_class
    num_classes = len(dataset.selected_classes)

    arr_targets = np.array(ds.targets)

    arr_victim_indices = ds.victim_indices

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
        if lvl > 0.0:
            assert len(set(arr_targets[arr_victim_indices].tolist())) == num_classes
        else:
            assert len(arr_victim_indices) == 0


@pytest.mark.parametrize(
    "train_split",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "variant",
    [
        "spurious-copyrightwatermarkjpeg",
    ],
)
@pytest.mark.parametrize(
    "dataset_name,lvl",
    [
        ("cat", 1.0),
        ("cat", 0.5),
        ("valsplit-cat", 0.0),
        ("valsplit-cat", 1.0),
    ],
)
@pytest.mark.slow
def test_dataset_with_three_spurious_correlations(
    lvl, train_split, dataset_name, variant
):

    if "valsplit" in dataset_name:
        atol = 60 if train_split else 30
    else:
        atol = 50 if train_split else 15


    dataset = datasets.construct(f"imagenet-{dataset_name}--{variant}--{lvl}")
    num_classes = len(dataset.selected_classes)
    ds: datasets.imagenet.TorchVisionDatasetImageNetWithThreeSpuriousFeatures = (
        dataset.create_subset(train_split=train_split)
    )

    num_victim_classes = num_classes - 1

    counts = np.zeros(4)
    for l in np.array(ds.arr_data_spurious).astype(int):
        counts[l] = counts[l] + 1

    if train_split:
        if "valsplit" in dataset_name:
            n_per_class = int(1300 * 0.8)
        else:
            n_per_class = 1300

        n_spurious_samples_per_class = np.floor(n_per_class * lvl)
        np.testing.assert_allclose(np.sum(counts), n_per_class * num_classes)

        np.testing.assert_allclose(
            counts,
            [
                (n_per_class - n_spurious_samples_per_class) * num_victim_classes
                + n_per_class,
                n_spurious_samples_per_class * 2,
                n_spurious_samples_per_class * 2,
                n_spurious_samples_per_class * 2,
            ],
            atol=atol,
        )
    else:
        if "valsplit" in dataset_name:
            # this is val set of valsplit
            n_per_class = int(1300 * 0.2)
        else:
            n_per_class = 50

        total = len(ds)

        n_spurious_samples_per_type = int(total * lvl) / 3

        np.testing.assert_allclose(
            counts,
            [
                total - 3 * n_spurious_samples_per_type,
                n_spurious_samples_per_type,
                n_spurious_samples_per_type,
                n_spurious_samples_per_type,
            ],
            atol=atol,
        )
