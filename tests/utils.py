import pytest

import torch
from torch import nn

from torch.nn import functional as F

import numpy as np
import torchvision

from xaikd import utils, models, datasets, constants


def test_subsample():
    np.random.seed(1)
    act = np.random.randn(10, 3, 5, 5)
    ctx = np.random.randn(10, 3, 5, 5)
    subsampled_act, subsampled_ctx = utils.subsample_tensors(act, ctx, num_locations=13)

    assert subsampled_act.shape == subsampled_ctx.shape
    assert subsampled_act.shape == (10, 3, 13)


def test_flatten():
    bs = 10
    wh = 20
    x = torch.randn(bs, 16, wh).numpy()

    flatten_x = utils.flatten_3d_tensor(x)

    for i in range(bs):
        for j in range(wh):

            pos = i * wh + j
            actual = flatten_x[pos]
            expected = x[i, :, j]
            np.testing.assert_allclose(actual, expected)


def test_count_params():
    lin1 = nn.Linear(20, 16)

    lin2 = nn.Linear(16, 7)

    utils.deactivate_requires_grad(lin2)

    model = nn.Sequential(lin1, lin2)

    total, trainable = utils.count_params_in_model(model)

    assert total == ((20 + 1) * 16 + (16 + 1) * 7)
    assert trainable == ((20 + 1) * 16)


def test_query_module_with_types():
    module = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=1),
        nn.BatchNorm2d(32),
        nn.Sequential(nn.Conv2d(32, 64, kernel_size=1), nn.BatchNorm2d(64)),
        nn.Conv2d(1, 64, kernel_size=1),
        nn.BatchNorm2d(64),
        nn.Linear(64, 10),
    )

    arr_batchnorm = utils.query_module_children_with_type(module, nn.BatchNorm2d)
    arr_conv2d = utils.query_module_children_with_type(module, nn.Conv2d)
    arr_linear = utils.query_module_children_with_type(module, nn.Linear)

    assert len(arr_batchnorm) == 3
    assert len(arr_conv2d) == 3
    assert len(arr_linear) == 1


def test_query_module_with_types_resnet18_cifar100():
    module = models.get_trained_model("cifar100-resnet18-v1")

    arr_batchnorm = utils.query_module_children_with_type(module, nn.BatchNorm2d)

    assert len(arr_batchnorm) == 20


def test_logspace():
    d = 512

    np.testing.assert_equal(
        [1] + np.logspace(1, 9, num=9, base=2).tolist(), utils.logspace(d)
    )

    d = 530

    np.testing.assert_equal(
        [1] + np.logspace(1, 9, num=9, base=2).tolist() + [d], utils.logspace(d)
    )


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("resnet18", dict(layer1=64, layer2=128, layer3=256, layer4=512)),
        # ("resnet50", dict(layer1=256, layer2=512, layer3=1024, layer4=2048)),
    ],
)
def test_get_dimensions(arch, expected):
    layers = expected.keys()
    dataset = datasets.construct("cifar100-people")

    dl = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False, batch_size=5
    )

    model = models.get_trained_model(f"cifar100-{arch}-v1")

    actual = utils.get_dimensions_at_layers(model, dl, layers=layers)

    np.testing.assert_equal(actual, expected)


@torch.no_grad()
def test_transformation_with_linear():
    feat = torch.randn(20, 30, 7, 7)

    linear = nn.Linear(in_features=30, out_features=10, bias=False)

    expected = F.conv2d(feat, linear.weight.unsqueeze(2).unsqueeze(3)).numpy()

    actual = utils.convolve_feature_map_with_linear(feat, linear).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@torch.no_grad()
def test_compute_log_odd_wining():
    logits = torch.tensor(
        [
            [5.0, 3],
            [2, 10],
        ]
    )

    p_y_gx_x = torch.softmax(logits, dim=1).detach().numpy()

    actual_log_odd = utils.compute_log_odd_winning(logits=logits).detach().numpy()

    p_winning = np.array(
        [
            p_y_gx_x[0, 0],
            p_y_gx_x[1, 1],
        ]
    )

    expected_log_odd = np.log(p_winning) - np.log(1 - p_winning)

    np.testing.assert_allclose(actual_log_odd, expected_log_odd, atol=1e-4)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        ("88", 88),
        ("layer3", None),
        ("-1", -1),
    ],
)
def test_parse_number_if_possible(text, expected):
    actual = utils.parse_number_if_possible(text)
    assert actual == expected


def test_solve_eigh():
    rng = np.random.default_rng(seed=1)
    arr_act = rng.integers(
        0,
        10,
        size=(10, 5),
    )
    arr_ctx = rng.integers(0, 10, size=(10, 5))

    # case 1: psd
    cov = arr_act.T @ arr_act

    actual_cov_eigvals, actual_cov_eigvecs = utils.solve_eigh(cov=cov)
    expected_cov_eigvals, expected_cov_eigvecs = np.linalg.eigh(cov)

    np.testing.assert_allclose(
        actual_cov_eigvals,
        np.flip(expected_cov_eigvals),
    )

    assert (actual_cov_eigvals >= 0).all()

    np.testing.assert_allclose(
        actual_cov_eigvecs,
        np.flip(expected_cov_eigvecs, axis=1),
    )

    # case 2: indefinite
    ccov = arr_act.T @ arr_ctx + arr_ctx.T @ arr_act

    # case 2.1: sort raw eigvals
    ccov_eigvals, ccov_eigvecs = np.linalg.eigh(ccov)
    assert (np.mean(ccov_eigvals >= 0) > 0) and (np.mean(ccov_eigvals < 0) > 0)

    actual_ccov_eigvals, actual_ccov_eigvecs = utils.solve_eigh(cov=ccov)
    np.testing.assert_allclose(actual_ccov_eigvals, np.flip(ccov_eigvals))
    np.testing.assert_allclose(actual_ccov_eigvecs, np.flip(ccov_eigvecs, axis=1))

    # case 2.2: sort abs eigvals
    actual_ccov_abs_eigvals, actual_ccov_abs_eigvecs = utils.solve_eigh(
        cov=ccov, sort_with_abs_eigvals=True
    )
    ccov_abs_eigvals = np.abs(ccov_eigvals)
    sorted_abs_idx = np.argsort(-ccov_abs_eigvals)

    expected_ccov_abs_eigvals = ccov_abs_eigvals[sorted_abs_idx]
    expected_ccov_abs_eigvecs = ccov_eigvecs[:, sorted_abs_idx]

    np.testing.assert_allclose(actual_ccov_abs_eigvals, expected_ccov_abs_eigvals)
    np.testing.assert_allclose(actual_ccov_abs_eigvecs, expected_ccov_abs_eigvecs)


@pytest.mark.slow()
def test_modify_last_layer_for_subclasses():
    device = utils.get_device()
    model = models.get_trained_model("cifar100-resnet18-v1").to(device)
    data = torch.randn(5, 3, 32, 32).to(device)

    with torch.no_grad():
        utils.modify_last_layer_for_subclasses(model, list(range(8)))
        output = model(data).cpu().numpy()
        assert output.shape == (5, 8)
