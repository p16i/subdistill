import pytest

import numpy as np

from torchvision.datasets import ImageNet

from xaikd import datasets


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
        train=train_split,
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
@pytest.mark.parametrize(
    "lvl",
    [
        0.0,
        0.5,
        1.0,
    ],
)
def test_dataset_accessible(dataset_name, lvl, expected_class_indices):

    arr_datasets = []
    if lvl > 0:
        for cix, _ in enumerate(expected_class_indices):
            arr_datasets.append(
                "--".join([dataset_name, f"spurious-threespurious", f"{lvl}"])
            )

    else:
        arr_datasets = [dataset_name]

    for dataset in arr_datasets:
        dataset = datasets.construct(dataset)

        np.testing.assert_array_equal(dataset.selected_classes, expected_class_indices)


@pytest.mark.skip(reason="obsolete")
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
        "spurious-threespurious",
    ],
)
@pytest.mark.parametrize(
    "dataset_name,lvl",
    [
        ("valsplit-cat", 0.0),
        ("valsplit-cat", 1.0),
        ("cat", 1.0),
        ("cat", 0.75),
        ("cat", 0.5),
        ("cat", 0.25),
        ("cat", 0.1),
        ("valsplit-butterfly", 1.0),
        ("butterfly", 1.0),
        ("butterfly", 0.5),
    ],
)
@pytest.mark.slow
def test_dataset_with_three_spurious_correlations(
    lvl, train_split, dataset_name, variant
):

    if "valsplit" in dataset_name:
        atol = 60 if train_split else 35
    else:
        atol = 50 if train_split else 17

    dataset = datasets.construct(f"imagenet-{dataset_name}--{variant}--{lvl}")
    num_classes = len(dataset.selected_classes)
    ds = dataset.create_subset(train_split=train_split)

    assert isinstance(
        ds,
        datasets.imagenet.subclasses.TorchVisionDatasetImageNetWithThreeSpuriousFeatures,
    )

    total_spurious_types = dataset.dataclass.total_spurious_types

    counts = np.zeros(dataset.dataclass.total_spurious_types)
    for l in np.array(ds.arr_data_spurious).astype(int):
        counts[l] = counts[l] + 1

    if train_split:
        if "valsplit" in dataset_name:
            n_per_class = int(1300 * constants.TRAINING_VAL_SPLIT_RATIO)

            np.testing.assert_allclose(
                len(ds), 1300 * num_classes * constants.TRAINING_VAL_SPLIT_RATIO, atol=2
            )
        else:
            n_per_class = 1300

        spurious_type_factors = np.bincount(
            np.arange(num_classes) % total_spurious_types
        ).astype(float)

        np.testing.assert_allclose(np.sum(counts), n_per_class * num_classes, atol=2)

        expected_counts = n_per_class * ((spurious_type_factors * lvl))

        # remark:  we take into account "samples" that are not contaminated.
        expected_counts += n_per_class * np.array(
            [(1 - lvl) * np.sum(spurious_type_factors), 0, 0, 0]
        )

        np.testing.assert_allclose(
            counts,
            expected_counts,
            atol=atol,
        )

        for cix, cls_ix in enumerate(sorted(dataset.selected_classes)):
            expected_spurious_type = cix % total_spurious_types if lvl > 0 else 0
            class_sample_indices = np.argwhere(np.array(ds.targets) == cls_ix).reshape(
                -1
            )
            np.testing.assert_allclose(
                class_sample_indices.shape[0], n_per_class, atol=atol
            )

            expected_lvl = lvl if expected_spurious_type != 0 else 1.0
            np.testing.assert_allclose(
                (
                    np.array(ds.arr_data_spurious)[class_sample_indices]
                    == expected_spurious_type
                ).mean(),
                expected_lvl,
                atol=1e-2,
                err_msg=f"cix={cix}; expected_type={expected_spurious_type} (shape: {class_sample_indices.shape})",
            )

    else:
        if "valsplit" in dataset_name:
            # this is val set of valsplit
            n_per_class = int(1300 * 0.2)
        else:
            n_per_class = 50

        total = len(ds)

        n_spurious_samples_per_type = int(total * lvl) / total_spurious_types

        expected_counts = [
            (total - (total_spurious_types - 1) * n_spurious_samples_per_type),
            n_spurious_samples_per_type,
            n_spurious_samples_per_type,
            n_spurious_samples_per_type,
        ]

        np.testing.assert_allclose(
            counts,
            expected_counts,
            atol=atol,
        )

        expected_count_spurious_type = n_per_class * (
            (lvl * np.ones(total_spurious_types) / total_spurious_types)
        )

        # remark:  we take into account "samples" that are not contaminated.
        expected_count_spurious_type += n_per_class * np.array([(1 - lvl), 0, 0, 0])

        for cix, cls_ix in enumerate(sorted(dataset.selected_classes)):
            expected_spurious_type = cix % total_spurious_types if lvl > 0 else 0
            class_sample_indices = np.argwhere(np.array(ds.targets) == cls_ix).reshape(
                -1
            )

            count_spurious_types = np.zeros(4)
            for six in class_sample_indices:
                spurious_type = int(ds.arr_data_spurious[six])
                count_spurious_types[spurious_type] += 1

            np.testing.assert_allclose(
                count_spurious_types, expected_count_spurious_type, atol=atol
            )


@torch.no_grad()
@pytest.mark.slow()
def test_construct_superclass_vs_others():
    dataset = datasets.construct("imagenet-butterfly-vs-others")

    num_classes = len(dataset.selected_classes)

    assert dataset.num_classes == 1
    assert num_classes == len(
        datasets.imagenet.IMAGENET_SUPERCLASS_MAPPING["butterfly"]
    )

    # remark: we use the val set and subsample it
    # to make the test run in reasonable time
    ds = dataset.create_subset(train_split=False)
    ds = datasets.subsample_dataset(ds, ratio=0.1, seed=1)

    arr_ys = []
    dl = datasets.build_dataloader(dataset=ds, shuffle=False)
    for _, y in tqdm(dl):
        arr_ys.extend(y.numpy().tolist())

    arr_ys = np.array(arr_ys)

    perc_y1 = (arr_ys == 1).mean()

    np.testing.assert_allclose(perc_y1, 6 / 1000, atol=1e-3)
