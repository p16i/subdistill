import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
import pytest

from xaikd import models, utils

from xaikd.utils.modules import (
    Centering2d,
    has_batchnorm,
    convert_bn_to_conv,
    merge_conv_and_bn,
    merge_convKxK_and_conv1x1,
    torch_flatten_3d_tensor,
    torch_deflatten_2d_tensor,
    CovarianceEigenspaceProjection,
)


@torch.no_grad()
@pytest.mark.parametrize("affine,dims,w", [(True, 10, 4), (False, 10, 4)])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_bn_to_conv(affine, dims, w, seed):
    torch.manual_seed(seed)

    bn = nn.BatchNorm2d(num_features=dims, affine=affine)

    if affine:
        bn.weight = nn.Parameter(torch.rand_like(bn.weight))
        bn.bias = nn.Parameter(torch.rand_like(bn.bias))

    # generate training data
    x_train = torch.randn(5, dims, w, w) + torch.randint(low=3, high=7, size=(1,))

    # update some statistics
    bn(x_train)

    # start convertion and testing
    bn.eval()

    # generate test data
    x_test = torch.rand(5, dims, w, w)
    expected = bn(x_test)

    conv_bn = convert_bn_to_conv(bn)

    actual = conv_bn(x_test).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,padding,w",
    [
        (5, 10, 3, 1, 7),
        (17, 8, 4, 0, 10),
    ],
)
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("bias", [True, False])
def test_merge_convK_and_conv1(
    in_channels, out_channels, kernel_size, padding, w, seed, bias
):
    torch.manual_seed(seed)

    x = torch.rand(16, in_channels, w, w)

    convK = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias,
    )

    conv1 = nn.Conv2d(
        in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0
    )

    expected = conv1(convK(x)).numpy()

    merged_conv = merge_convKxK_and_conv1x1(convK, conv1)
    actual = merged_conv(x).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,padding,w",
    [
        (5, 10, 3, 1, 7),
        (17, 8, 4, 0, 10),
    ],
)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_merge_conv_and_bn(
    in_channels, out_channels, kernel_size, padding, w, affine, seed
):
    torch.manual_seed(seed)

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
    )

    bn = nn.BatchNorm2d(num_features=out_channels, affine=affine)

    if affine:
        bn.weight = nn.Parameter(torch.rand_like(bn.weight))
        bn.bias = nn.Parameter(torch.rand_like(bn.bias))

    x_train = torch.rand(16, in_channels, w, w)

    # mock forward pass of training data
    bn(conv(x_train))

    # inference pass
    x_test = torch.rand(16, in_channels, w, w)
    bn.eval()

    expected = bn(conv(x_test)).numpy()

    merged_conv = merge_conv_and_bn(conv, bn)

    actual = merged_conv(x_test).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("imagenet-vgg11-tv", False),
        ("imagenet-vgg11bn-tv", True),
        ("imagenet-vgg16-tv", False),
        ("imagenet-vgg16bn-tv", True),
        ("imagenet-resnet18-tv", True),
    ],
)
def test_has_batchnorm(arch, expected):
    actual = has_batchnorm(models.get_trained_model(arch))

    assert actual == expected


def test_centering2d():
    torch.manual_seed(1)
    x = torch.randn(100, 20, 7, 7)

    mean = x.mean(dim=(0, 2, 3))

    centering = Centering2d(20)
    centering.train()

    actual = centering(x)

    expected = x - mean.reshape(1, -1, 1, 1)

    assert isinstance(centering.running_mean, torch.Tensor)

    # check training mode
    np.testing.assert_allclose(
        centering.running_mean.numpy(), mean.squeeze().numpy(), atol=0.1
    )
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=0.1)

    # check eval mode
    centering.eval()
    centering(torch.randn(100, 20, 7, 7) + 10)

    np.testing.assert_allclose(
        centering.running_mean.numpy(), mean.squeeze().numpy(), atol=0.1
    )


def test_torch_flatten():
    expected = x = torch.randn(7, 5, 3, 3)

    original_shape = x.shape

    actual = torch_deflatten_2d_tensor(
        torch_flatten_3d_tensor(x), target_shape=original_shape
    )

    np.testing.assert_allclose(actual, expected)


@torch.no_grad()
def test_coveigen_forwardble():
    d = 3
    n = 5

    trng = torch.Generator().manual_seed(1)

    module = CovarianceEigenspaceProjection(num_features=d)

    momentum = module.momentum

    expected_mean = torch.zeros(d)
    expected_cov = torch.zeros((d, d))
    expected_eigvecs = torch.eye(d)

    for i, training in enumerate([True, False]):
        x = torch.randn(n, d, 7, 7, generator=trng)

        assert x.shape == (n, d, 7, 7)

        x_flattened = torch.flatten(torch.permute(x, (1, 0, 2, 3)), start_dim=1)
        if training:
            module.train()

            expected_mean = x_flattened.mean(dim=1).numpy()
            expected_cov = torch.cov(x_flattened).numpy()

            _, expected_eigvecs = utils.solve_eigh(expected_cov)

            expected_eigvecs = utils.modules.adjust_basis_vectors_to_positive_direction(
                torch.from_numpy(expected_eigvecs), x_flattened.T
            ).numpy()
        else:
            module.eval()

        actual_output = module(x)

        np.testing.assert_allclose(
            module.running_mean.numpy(), momentum * expected_mean, atol=1e-5
        )
        np.testing.assert_allclose(
            module.running_cov.numpy(), momentum * expected_cov, atol=1e-5
        )

        np.testing.assert_allclose(
            # abs() takes into account ambiguity of sign
            np.abs((module.running_eigvecs.numpy().T @ expected_eigvecs)),
            np.eye(d),
            atol=1e-5,
        )

        if training:
            expected_output = x_flattened.T - expected_mean
        else:
            expected_output = x_flattened.T - momentum * expected_mean

        expected_output = expected_output @ expected_eigvecs

        assert expected_output.shape == x_flattened.T.shape
        expected_output_reshaped = torch.permute(
            expected_output.T.reshape(d, n, 7, 7), (1, 0, 2, 3)
        )

        assert actual_output.shape == expected_output_reshaped.shape

        np.testing.assert_allclose(actual_output, expected_output_reshaped, atol=1e-3)


def test_adjust_direction_torch():
    d = 7
    x = torch.rand(10, d, generator=torch.Generator().manual_seed(1))

    U = -torch.eye(d)

    np.testing.assert_allclose(
        utils.modules.adjust_basis_vectors_to_positive_direction(U, x).numpy(),
        np.eye(d),
        atol=1e-6,
    )
