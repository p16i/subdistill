import pytest
import numpy as np

import numpy.typing as npt


from xaikd import bases


@pytest.mark.parametrize("basis_name", ["pca", "prca-sortabs", "prca"])
@pytest.mark.parametrize("suffix", ["centered", "uncentered"])
def test_analytic_basis(basis_name, suffix):
    rng = np.random.default_rng(seed=1)
    n, d = 10, 4

    arr_act = rng.random(size=(n, d)) + 2
    arr_ctx = rng.random(size=(n, d)) + 2

    mean = arr_act.mean(axis=0)

    if suffix == "centered":
        arr_modified_act = arr_act - mean
    else:
        arr_modified_act = arr_act

    basis = bases.get_basis(f"{basis_name}--{suffix}")
    print(f"centerning={basis.centering}")

    basis.fit(arr_act, arr_ctx)

    # compute expected U
    if basis_name == "pca":
        expected_U = np.flip(
            np.linalg.eigh(arr_modified_act.T @ arr_modified_act)[1], axis=1
        )
    elif basis_name in ["prca", "prca-sortabs"]:
        eigvals, eigvecs = np.linalg.eigh(
            arr_modified_act.T @ arr_ctx + arr_ctx.T @ arr_modified_act
        )
        print(f"Ttest: eigvals: {eigvals}")

        if basis_name == "prca":
            expected_U = np.flip(eigvecs, axis=1)
        elif basis_name == "prca-sortabs":
            expected_U = eigvecs[:, np.argsort(-np.abs(eigvals))]
        else:
            raise
    else:
        raise
    # end

    # verification
    if suffix == "centered":
        np.testing.assert_allclose(basis.mean, mean)
    else:
        np.testing.assert_allclose(basis.mean, np.zeros(d))

    np.testing.assert_allclose(basis.U, expected_U)

    np.testing.assert_allclose(
        basis.scale, np.mean((arr_modified_act @ basis.U) ** 2, axis=0)
    )


# @pytest.mark.parametrize("basis_mode", ["centered", "uncentered"])
# @pytest.mark.parametrize(
#     "basis_name,criteria",
#     [
#         ("prca", lambda eigvals: eigvals),
#         ("prca-sortabs", lambda eigvals: np.abs(eigvals)),
#     ],
# )
# def test_prca(basis_name, basis_mode, criteria):
#     np.random.seed(1)
#     n, d = 10, 5

#     activation = np.random.randn(n, d) + 2
#     context = np.random.rand(n, d)

#     if basis_mode == "centered":
#         mean = activation.mean(axis=0)
#     else:
#         mean = np.zeros(d)

#     centered_activation = activation - mean

#     crosscov = ((centered_activation.T @ context) + context.T @ centered_activation) / n

#     eigvals, eigvecs = np.linalg.eigh(crosscov)

#     basis = bases.get_basis(f"{basis_name}--{basis_mode}")

#     U, scale = basis.fit(activation, context, mean=mean, device="cpu")

#     expected_U = eigvecs[:, np.argsort(-criteria(eigvals))]

#     np.testing.assert_allclose(U, expected_U)

#     np.testing.assert_allclose(scale, np.mean((centered_activation @ U) ** 2, axis=0))

#     assert f"{basis}" == f"{basis_name}--{basis_mode}"


# @pytest.mark.parametrize("variant", ["abs", "recon", "reconreg0.1"])
# @pytest.mark.parametrize("slug", ["centered"])
# def test_instantiate_prca_greedy_basese(variant, slug):
#     basis = bases.get_basis(f"prca-{variant}--{slug}")
#     assert True


# @pytest.mark.parametrize(
#     "basis_name",
#     [
#         "pca",
#         "prca-abs",
#         "prca",
#         "prca-sortabs",
#         "prca-recon",
#         "act-recon",
#         "rel-recon",
#         "rel-raw",
#         "random",
#     ],
# )
# @pytest.mark.parametrize("basis_mode", ["centered", "uncentered"])
# def test_correct_scale(basis_name, basis_mode):
#     np.random.seed(1)
#     n, d = 10, 5
#     activation = np.random.randn(n, d)
#     context = np.random.randn(n, d)

#     mean = activation.mean(axis=0)

#     modified_activation = activation - mean if basis_mode == "centered" else activation

#     basis = bases.get_basis(f"{basis_name}--{basis_mode}", seed=1)

#     eigvecs, scale = basis.fit(
#         activation=activation, context=context, mean=mean, device="cpu"
#     )

#     np.testing.assert_allclose(
#         scale, np.mean((modified_activation @ eigvecs) ** 2, axis=0)
#     )


# @pytest.mark.parametrize(
#     "basis_name,mat_func,criteria",
#     [
#         ("pca", lambda d: d[0].T @ d[0], lambda x: x),
#         ("prca", lambda d: d[0].T @ d[1] + d[1].T @ d[0], lambda x: x),
#         ("prca-sortabs", lambda d: d[0].T @ d[1] + d[1].T @ d[0], lambda x: np.abs(x)),
#     ],
# )
# @pytest.mark.parametrize("basis_mode", ["centered", "uncentered"])
# def test_centering(basis_name, mat_func, criteria, basis_mode):
#     np.random.seed(1)
#     n, d = 10, 5
#     activation = np.random.randn(n, d)
#     context = np.random.randn(n, d)

#     mean = np.mean(activation, axis=0)

#     modified_activation = activation - mean if basis_mode == "centered" else activation

#     expected_eigvals, expected_eigvecs = np.linalg.eigh(
#         mat_func((modified_activation, context))
#     )

#     expected_U = expected_eigvecs[:, np.argsort(-criteria(expected_eigvals))]

#     basis = bases.get_basis(f"{basis_name}--{basis_mode}", seed=1)

#     actual_U, _ = basis.fit(
#         activation=activation, context=context, mean=mean, device="cpu"
#     )

#     np.testing.assert_allclose(actual_U, expected_U)
