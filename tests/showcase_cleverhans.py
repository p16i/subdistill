import pytest
import numpy as np

import numpy.typing as npt

from xaikd import datasets
from xaikd.showcases import cleverhans


@pytest.mark.parametrize("contamination_level", [0.1, 0.3])
@pytest.mark.parametrize("training_size", [0.1, 0.5])
@pytest.mark.parametrize("dataset_name", ["cifar100-people", "cifar100"])
def test_contamination_dataset(dataset_name, contamination_level, training_size):
    dataset = datasets.construct(dataset_name)

    ds_clean = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=training_size, seed=1
    )

    ds_contaminated = cleverhans.contaminate_dataset(
        ds_clean, contamination_level=contamination_level, seed=3
    )

    subsampled_indices = ds_clean.indices
    targets = np.array(ds_clean.dataset.targets)

    assert id(ds_contaminated.dataset) != id(ds_clean.dataset)
    np.testing.assert_equal(ds_contaminated.indices, ds_clean.indices)

    # remark: this convention is used in the function.
    victim_class_ix = np.min(targets)

    all_possible_victim_samples_indices = np.argwhere(
        targets == victim_class_ix
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
        clean_data: npt.NDArray = ds_clean.dataset.data[index_set, :]
        contaminated_data: npt.NDArray = ds_contaminated.dataset.data[index_set, :]
        assert clean_data.shape[0] == contaminated_data.shape[0]
        n = clean_data.shape[0]

        ratio_equal_data_points = np.prod(
            clean_data.reshape((n, -1)) == contaminated_data.reshape((n, -1)), axis=1
        ).mean()

        np.testing.assert_allclose(1 - ratio_equal_data_points, expected_ratio)
