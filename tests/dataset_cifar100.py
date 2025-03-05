from collections import OrderedDict

import pytest

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch import nn
from torchvision.datasets import CIFAR100

import pandas as pd
from copy import deepcopy

from xaikd import utils, datasets, constants

DF_CIFAR100_LABEL_MAPPING = pd.read_csv(
    datasets.constants.PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv"
)


@pytest.mark.parametrize(
    "name",
    [
        "cifar100",
        "cifar100-people",
        # todo: the imagenet parameter should be with dataset_imagenet
        "imagenet",
    ],
)
def test_construct_dataset(name):
    dataset = datasets.construct(name)

    # todo: find a way to use @property to automatically validate these attributes
    # instead of do this manually
    assert hasattr(dataset, "num_classes")
    assert hasattr(dataset, "input_transformation")
    assert hasattr(dataset, "input_training_transformation")
    assert hasattr(dataset, "_normalizer")


@pytest.mark.parametrize(
    "super_class", DF_CIFAR100_LABEL_MAPPING["coarse_label_name"].unique().tolist()
)
def test_cifar100_superclass(super_class):
    ds: datasets.Cifar100SuperClassesDataset = datasets.construct(
        f"cifar100-{super_class}"
    )

    df = DF_CIFAR100_LABEL_MAPPING

    fine_labels = df[df.coarse_label_name == super_class].fine_label.values.tolist()

    assert tuple(sorted(ds.selected_classes)) == tuple(sorted(fine_labels))

    for ix, (_, y) in enumerate(DataLoader(ds.create_subset(train_split=False))):
        assert (y.numpy() <= ds.num_classes - 1).all()


@pytest.mark.parametrize(
    "super_class", DF_CIFAR100_LABEL_MAPPING["coarse_label_name"].unique().tolist()
)
def test_cifar100_superclass_transform_target(super_class):
    ds: datasets.Cifar100SuperClassesDataset = datasets.construct(
        f"cifar100-{super_class}"
    )

    df = DF_CIFAR100_LABEL_MAPPING

    fine_labels = sorted(
        df[df.coarse_label_name == super_class].fine_label.values.tolist()
    )

    dl_val = datasets.build_dataloader(
        ds.create_subset(train_split=False), shuffle=False
    )

    arr_ys = []
    for _, y in dl_val:
        arr_ys.extend(y.numpy().tolist())

    assert np.isin(arr_ys, range(len(fine_labels))).all()


@torch.no_grad()
@pytest.mark.slow
def test_target_transform():
    dataset_name = "cifar100-people"
    dataset = datasets.construct(dataset_name)

    model = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(3, 30, kernel_size=3)),
                ("pool1", nn.AdaptiveAvgPool2d(1)),
                ("flatten", nn.Flatten(start_dim=1)),
                ("__last_layer", nn.Linear(30, 100)),
            ]
        )
    )

    cloned_model = deepcopy(model)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)

    actual_dl_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    actual_x, actual_y = next(iter(actual_dl_val))
    assert np.isin(actual_y.numpy(), range(len(dataset.selected_classes))).all()

    actual_logits = model(actual_x)

    actual_acc = np.mean(np.argmax(actual_logits) == actual_y.numpy())

    actual_x, actual_y = next(iter(actual_dl_val))

    expected_all_logits, expected_all_y = [], []

    for x, y in DataLoader(
        CIFAR100(
            root=str(datasets.DATADIR / "cifar100"),
            train=False,
            transform=dataset.input_transformation,
        ),
        shuffle=False,
        batch_size=64,
    ):
        expected_all_logits.append(cloned_model(x).numpy())
        expected_all_y.append(y.numpy())

    expected_all_logits = np.vstack(expected_all_logits)
    expected_all_y = np.concatenate(expected_all_y)

    expected_indices = np.argwhere(
        np.isin(expected_all_y, dataset.selected_classes)
    ).reshape(-1)[: actual_logits.shape[0]]

    expected_logits = expected_all_logits[expected_indices]

    expected_acc = np.mean(
        np.argmax(expected_logits) == expected_all_y[expected_indices]
    )

    np.testing.assert_allclose(actual_acc, expected_acc)

    np.testing.assert_allclose(
        actual_logits, expected_logits[:, dataset.selected_classes], atol=1e-6
    )


@pytest.mark.parametrize("lvl", [0.125, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("train_split", [True, False])
@pytest.mark.parametrize("dataset_slug", ["cifar100-people--spurious-plussign"])
def test_dataset_with_spurious_correlation(
    dataset_slug, lvl, train_split, victim_class=None
):

    dataset = datasets.construct("--".join([dataset_slug, str(lvl)]))

    num_classes = len(dataset.selected_classes)

    if victim_class is None:
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
        if lvl > 0.0:
            assert len(set(arr_targets[arr_victim_indices].tolist())) == num_classes
        else:
            assert len(arr_victim_indices) == 0


@pytest.mark.parametrize("lvl", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("train_split", [True, False])
def test_valsplit_dataset_with_spurious_correlation(lvl, train_split):
    # testing the size of the split
    dataset = datasets.construct(f"cifar100-valsplit-people--spurious-plussign--{lvl}")
    # remark: here, we get subset of the official training set
    ds = dataset.create_subset(train_split=train_split)

    assert ds.train

    assert isinstance(ds, CIFAR100)

    dl = datasets.build_dataloader(ds, batch_size=int(1e6), shuffle=False)

    batch, _ = next(iter(dl))

    np.testing.assert_allclose(
        batch.shape[0],
        500
        * 5
        * (
            constants.TRAINING_VAL_SPLIT_RATIO
            if train_split
            else 1 - constants.TRAINING_VAL_SPLIT_RATIO
        ),
    )

    victim_class = dataset.selected_classes[0]
    num_classes = len(dataset.selected_classes)

    # this is global targets
    arr_targets = np.array(ds.targets)

    arr_victim_indices = ds.victim_indices
    num_samples = ds.data.shape[0]

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


@torch.no_grad()
@pytest.mark.parametrize(
    "dataset_name", ["cifar100-people-vs-others", "cifar100val-people-vs-others"]
)
def test_construct_superclass_vs_others(dataset_name):
    dataset = datasets.construct(dataset_name)

    assert len(dataset.selected_classes) == 5
    assert dataset.num_classes == 1

    for train_split in [True, False]:
        ds = dataset.create_subset(train_split=train_split)

        arr_ys = []
        dl = datasets.build_dataloader(dataset=ds, shuffle=False)
        for _, y in dl:
            arr_ys.extend(y.numpy().tolist())

        arr_ys = np.array(arr_ys)

        perc_y1 = (arr_ys == 1).mean()

        np.testing.assert_allclose(
            perc_y1, 1 / len(constants.CIFAR100_SUPER_CLASSES), atol=1e-2
        )


def test_get_fineclass_indices():
    actual_names, actual_idx = (
        datasets.cifar100.get_fineclass_names_indices_of_superclass("people")
    )

    expected_names = [
        "baby",
        "boy",
        "girl",
        "man",
        "woman",
    ]
    expected_idx = sorted([11, 98, 35, 2, 46])
    np.testing.assert_equal(actual_idx, expected_idx)
    np.testing.assert_equal(actual_names, expected_names)
