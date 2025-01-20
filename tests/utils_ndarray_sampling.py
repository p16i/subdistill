import pytest
import torch

import numpy as np


from xaikd import utils


def test_gather_locations_along_axis():

    num_locations = 4

    rng = np.random.default_rng(seed=1)

    x = rng.normal(size=(20, 3, 16, 16))

    for axis in [-1, -2]:

        arr_locations = utils.ndarray_subsampling.get_random_permutations(
            rng=rng, n=x.shape[0], total=x.shape[axis], num_chosen=num_locations
        )

        actual = utils.ndarray_subsampling.gather_locations_along_axis(
            x, axis=axis, arr_selected_locations=arr_locations
        )

        expected_shape = list(x.shape)
        expected_shape[axis] = num_locations
        np.testing.assert_equal(actual.shape, expected_shape)

        location_shape = [1] * 3

        location_shape[axis] = num_locations

        for i in range(x.shape[0]):

            expected = np.take_along_axis(
                x[i],
                arr_locations[i].reshape(location_shape),
                axis,
            )

            np.testing.assert_allclose(actual[i, :, :, :], expected)


def test_gather_locations_along_axes():

    axes = [-1, -2]
    arr_num_locations = (4, 6)

    rng = np.random.default_rng(seed=1)

    x = rng.normal(size=(20, 3, 16, 16))

    nb = x.shape[0]

    arr_locations = []
    for axis, num_locations in zip(axes, arr_num_locations):
        arr_locations.append(
            utils.ndarray_subsampling.get_random_permutations(
                rng=rng, n=x.shape[0], total=x.shape[axis], num_chosen=num_locations
            )
        )

    actual = utils.ndarray_subsampling.gather_locations_along_axes(
        x, axes=axes, arr_selected_locations=arr_locations
    )

    indices = np.argsort(axes)

    # normalized locations (lowest axis comes first)
    arr_normalized_num_locations = np.array(arr_num_locations)[indices].tolist()

    np.testing.assert_allclose(actual.shape, (20, 3, *arr_normalized_num_locations))

    for i in range(x.shape[0]):

        expected = x[i]

        for aix in range(len(axes)):
            indices = arr_locations[aix]
            axis = axes[axis]

            location_shape = [1] * 4

            location_shape[0] = nb
            location_shape[axis] = arr_num_locations[aix]
            indices = indices.reshape(location_shape)[i]

            expected = np.take_along_axis(expected, indices=indices, axis=axis)

        np.testing.assert_allclose(actual[i, :, :, :], expected)
