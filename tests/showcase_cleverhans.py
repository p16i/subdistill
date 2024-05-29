import pytest
import numpy as np

import numpy.typing as npt

from xaikd import datasets
from xaikd.showcases import cleverhans

from torch.utils.data import random_split

import torch


pytest.skip(allow_module_level=True)


@pytest.mark.parametrize("contamination_level", [0.0, 0.1, 0.3])
@pytest.mark.parametrize("training_size", [0.1, 0.5])
@pytest.mark.parametrize(
    "dataset_name,victim_class_indices",
    [
        ("cifar100-people", [2]),
        ("cifar100-people", [2, 46]),
        ("cifar100-people", [2, 11, 35, 46, 98]),
        ("cifar100", [0]),
        ("cifar100", list(range(10))),
    ],
)
def test_contamination_dataset(
    dataset_name, victim_class_indices, contamination_level, training_size
):
    dataset = datasets.construct(dataset_name)

    ds_clean = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=training_size, seed=1
    )

    targets = np.array(ds_clean.dataset.targets)

    ds_contaminated = cleverhans.contaminate_dataset(
        ds_clean,
        contamination_level=contamination_level,
        seed=3,
        victim_class_indices=victim_class_indices,
    )

    subsampled_indices = ds_clean.indices

    assert id(ds_contaminated.dataset) != id(ds_clean.dataset)
    np.testing.assert_equal(ds_contaminated.indices, ds_clean.indices)

    all_possible_victim_samples_indices = np.argwhere(
        np.isin(targets, victim_class_indices)
    ).reshape(-1)

    potential_victim_samples_indices = np.array(
        list(set(subsampled_indices).intersection(all_possible_victim_samples_indices))
    )
    non_victim_samples_indices = np.array(
        list(set(subsampled_indices).difference(all_possible_victim_samples_indices))
    )
    for index_set, expected_ratio in [
        (potential_victim_samples_indices, contamination_level),
        (non_victim_samples_indices, 0.0),
    ]:
        if len(index_set) == expected_ratio == 0:
            # this is the case that have contaminate all classes in the dataset
            # therefore, we do NOT have non victim samples.
            continue

        clean_data: npt.NDArray = ds_clean.dataset.data[index_set, :]
        contaminated_data: npt.NDArray = ds_contaminated.dataset.data[index_set, :]
        assert clean_data.shape[0] == contaminated_data.shape[0]
        n = clean_data.shape[0]

        ratio_equal_data_points = np.prod(
            clean_data.reshape((n, -1)) == contaminated_data.reshape((n, -1)), axis=1
        ).mean()

        np.testing.assert_allclose(1 - ratio_equal_data_points, expected_ratio)


@pytest.mark.parametrize("contamination_level", [0.1, 0.3, 0.5])
def test_contamination_with_val_split(contamination_level):
    dataset = datasets.construct("cifar100-people")
    victim = dataset.selected_classes

    ds_main = dataset.create_subset(train_split=True)

    ds_train, ds_val = random_split(
        ds_main, [0.8, 0.2], torch.Generator().manual_seed(1)
    )

    ds_train = cleverhans.contaminate_dataset(
        ds_train,
        contamination_level=contamination_level,
        seed=1,
        victim_class_indices=victim,
    )

    ds_val = cleverhans.contaminate_dataset(
        ds_val,
        contamination_level=contamination_level,
        seed=1,
        victim_class_indices=victim,
    )

    assert id(ds_main.data) != id(ds_train.dataset.data) != id(ds_val.dataset.data)

    total = ds_main.data.shape[0]

    all_indices = list(range(total))
    for subset in [ds_train, ds_val]:
        subset_indices = subset.indices
        total_subset = len(subset_indices)
        other_indices = list(set(all_indices).difference(subset_indices))

        ds_main.data[subset.indices,]

        # Other sampls that do NOT belon to the subset shoul NOT be affected
        # by the contamination.
        np.testing.assert_equal(
            ds_main.data[other_indices, :], subset.dataset.data[other_indices, :]
        )

        # The proportion of the affected sampls should approximately
        # be equal to the contamination level.
        diff = (
            (ds_main.data[subset_indices, :] != subset.dataset.data[subset_indices, :])
            .reshape(total_subset, -1)
            .sum(axis=1)
        )

        np.testing.assert_allclose(
            (diff > 0).mean(),
            contamination_level,
        )
