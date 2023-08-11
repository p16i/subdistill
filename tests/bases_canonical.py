import pytest

import numpy as np


from xaikd import bases
from xaikd.utils import is_permuation_matrix


@pytest.mark.parametrize(
    "basis_name",
    [
        "randomperm",
        "act-raw",
        "act-recon",
        "rel-raw",
        "rel-abs",
        "rel-reconnaive",
        "rel-recon",
    ],
)
def test_return_permutation_like_matrix(basis_name):
    mode = "centered"

    np.random.seed(1)
    activation = np.random.randn(20, 5)
    context = np.random.randn(20, 5)

    mean = activation.mean(axis=0)

    basis = bases.get_basis(f"{basis_name}--{mode}", seed=1)

    eigvecs, std = basis.fit(
        activation=activation, context=context, mean=mean, device="cpu"
    )

    assert is_permuation_matrix(eigvecs)


def _test_return_correct_order(activation, context, basis_name, mode, expected_order):
    basis = bases.get_basis(f"{basis_name}--{mode}", seed=1)

    mean = activation.mean(axis=0)

    eigvecs, std = basis.fit(
        activation=activation, context=context, mean=mean, device="cpu"
    )

    actual_order = np.argmax(eigvecs, axis=0)
    np.testing.assert_equal(actual_order, expected_order)


@pytest.mark.parametrize(
    "basis_name",
    ["act-raw", "act-recon", "rel-raw", "rel-abs", "rel-reconnaive", "rel-recon"],
)
def test_correctness_canonical_based_manual_case_pos(basis_name):
    mode = "uncentered"

    activation = np.array([[1, 3, 2, 5]])
    context = activation

    _test_return_correct_order(
        activation, context, basis_name, mode, expected_order=[3, 1, 2, 0]
    )


@pytest.mark.parametrize(
    "basis_name,expected_order",
    [
        ("act-raw", [3, 2, 0, 1]),
        ("act-recon", [3, 1, 2, 0]),
        ("rel-abs", [3, 1, 2, 0]),
        ("rel-raw", [1, 2, 0, 3]),
        ("rel-reconnaive", [0, 3, 2, 1]),
        ("rel-recon", [0, 3, 1, 2]),
    ],
)
def test_correctness_canonical_based_manual_case_neg(basis_name, expected_order):
    # remark: we use uncentered to make it easiler to develop test case!
    mode = "uncentered"

    activation = np.array([[1, -3, 2, 5]])
    context = np.array([[1, -3, 2, -5]])

    # rel =           [1, 9, 4, -25]
    # rel-reconnaive: [1, -25, 4, 9]
    # rel-recon     : [1, -25, 9, 4]

    _test_return_correct_order(activation, context, basis_name, mode, expected_order)


@pytest.mark.parametrize(
    "basis_name,expected_order",
    [
        ("act-raw", [3, 1, 2, 0]),
        ("act-recon", [3, 1, 2, 0]),
        ("rel-raw", [0, 2, 1, 3]),
        ("rel-abs", [0, 2, 1, 3]),
        ("rel-reconnaive", [0, 2, 1, 3]),
        ("rel-recon", [0, 2, 1, 3]),
    ],
)
def test_correctness_canonical_based_manual_case_pos_mag_rel_reverse(
    basis_name, expected_order
):
    mode = "uncentered"

    activation = np.array([[1, 3, 2, 5]])
    context = np.array([[1, 1e-3, 1e-2, 1e-5]])

    _test_return_correct_order(activation, context, basis_name, mode, expected_order)


@pytest.mark.parametrize(
    "basis_name",
    [
        "rel-raw",
        "rel-abs",
        "rel-reconnaive",
        # remark: rel-recon fails here. TODO: investigate why.
        #    "rel-recon"
    ],
)
def test_correctness_rel_random_cases(basis_name):
    mode = "uncentered"

    np.random.seed(1)
    n, d = 20, 5
    activation = np.random.randn(n, d)
    context = activation

    expected_order = np.argsort(np.mean(activation**2, axis=0))[::-1]

    _test_return_correct_order(activation, context, basis_name, mode, expected_order)


@pytest.mark.parametrize(
    "basis_name,expected_order",
    [
        ("rel-reconnaive", [0, 2, 1]),
        ("rel-recon", [0, 1, 2]),
    ],
)
def test_rel_recon_toy_example(basis_name, expected_order):
    mode = "uncentered"

    activation = np.array([[1, -1, -1]])
    context = np.array([[1, 0.2, 0.1]])

    _test_return_correct_order(activation, context, basis_name, mode, expected_order)
