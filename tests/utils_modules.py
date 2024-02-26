import torch
from torch import nn

import numpy as np
import pytest

from xaikd.utils.modules import (
    convert_bn_to_conv,
    merge_conv_and_bn,
    merge_convKxK_and_conv1x1,
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
