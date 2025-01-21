import typing
import numpy as np
from numpy import typing as npt


def get_random_permutations(
    rng: np.random.Generator, n: int, total: int, num_chosen: int, verbose=False
):

    if num_chosen > total:
        if verbose:
            print(
                f"[warning] num_chosen ({num_chosen}) > total ({total}); effectively, we only permutate locations."
            )
        num_chosen = total

    candidates = np.arange(total)

    arr_selected = []
    for _ in range(n):
        arr_selected.append(rng.permutation(candidates)[:num_chosen])

    arr_selected = np.array(arr_selected)

    np.testing.assert_equal(arr_selected.shape, (n, num_chosen))

    return arr_selected


def gather_locations_along_axis(
    x: npt.NDArray,
    axis: int,
    arr_selected_locations: npt.NDArray,
) -> npt.NDArray:

    nb = x.shape[0]

    assert arr_selected_locations.shape[0] == nb
    _, num_locations = arr_selected_locations.shape

    shape = [1] * len(x.shape)
    shape[0] = nb
    shape[axis] = num_locations

    arr_selected_locations = arr_selected_locations.reshape(shape)
    output = np.take_along_axis(x, indices=arr_selected_locations, axis=axis)

    return output


def gather_locations_along_axes(
    x: npt.NDArray,
    axes: typing.List,
    arr_selected_locations: npt.NDArray,
) -> npt.NDArray:

    for axis, locations in zip(axes, arr_selected_locations):
        x = gather_locations_along_axis(
            x=x,
            axis=axis,
            arr_selected_locations=locations,
        )

    return x
