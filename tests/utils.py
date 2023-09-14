import pytest

from torch import nn

import numpy as np

from xaikd import utils
from xaikd import models
from xaikd import datasets


def test_subsample():
    np.random.seed(1)
    act = np.random.randn(10, 3, 5, 5)
    ctx = np.random.randn(10, 3, 5, 5)
    subsampled_act, subsampled_ctx = utils.subsample_tensors(act, ctx, num_locations=13)

    assert subsampled_act.shape == subsampled_ctx.shape
    assert subsampled_act.shape == (10 * 13, 3)


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
    module = models.get_trained_model("cifar100-resnet18-p1")

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
        ("resnet50", dict(layer1=256, layer2=512, layer3=1024, layer4=2048)),
    ],
)
def test_get_dimensions(arch, expected):
    layers = expected.keys()
    dataset = datasets.construct("cifar100-people")

    dl = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False, batch_size=5
    )

    model = models.get_trained_model(f"cifar100-{arch}-p1")

    actual = utils.get_dimensions_at_layers(model, dl, layers=layers)

    np.testing.assert_equal(actual, expected)
