import pytest
import numpy as np

import numpy.typing as npt

from xaikd import datasets
from xaikd.showcases import cleverhans


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
