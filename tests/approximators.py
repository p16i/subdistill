import numpy as np
import pytest
import torch

from xaikd import approximators, models
from xaikd.utils import count_params_in_model


@pytest.mark.parametrize(
    "model,layer",
    [
        ("cifar100-resnet18-p1", "layer3"),
        ("cifar100-resnet18-p1", "layer4"),
    ],
)
def test_approximator_homogenous_mode(model, layer):
    model = models.get_model(model)

    approx = approximators.construct_approximator_for(
        model=model,
        layer=layer,
        compression_ratio=1.0,
        mode=approximators.ApproximatorMode.HOMOGENOUS,
        seed=1,
    )

    _, actual_trainable_params = count_params_in_model(approx)
    _, expected_trainable_params = count_params_in_model(getattr(model, layer))

    assert actual_trainable_params == expected_trainable_params

    setattr(model, layer, approx)

    with torch.no_grad():
        y = model(torch.randn((10, 3, 32, 32)))

    assert not torch.isnan(y).all()


@pytest.mark.parametrize(
    "model,layer",
    [
        ("cifar100-resnet18-p1", "layer3"),
        ("cifar100-resnet18-p1", "layer4"),
    ],
)
def test_approximator_homogenous_lowrank_modes(model, layer):
    model = models.get_model(model)
    compression_rate = 4.0

    approx_lowrank = approximators.construct_approximator_for(
        model=model,
        layer=layer,
        compression_ratio=compression_rate,
        mode=approximators.ApproximatorMode.HOMOGENOUS_LOWRANK,
        seed=1,
    )

    approx_lowrank_adapter = approximators.construct_approximator_for(
        model=model,
        layer=layer,
        compression_ratio=compression_rate,
        mode=approximators.ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER,
        seed=1,
    )

    _, trainable_params_lowrank = count_params_in_model(approx_lowrank)
    _, trainable_params_lowrank_with_adapter = count_params_in_model(
        approx_lowrank_adapter
    )
    assert trainable_params_lowrank < trainable_params_lowrank_with_adapter

    setattr(model, layer, approx_lowrank_adapter)

    with torch.no_grad():
        y = model(torch.randn((10, 3, 32, 32)))

    assert not torch.isnan(y).all()


@torch.no_grad()
@pytest.mark.parametrize(
    "model,layer",
    [
        ("cifar100-resnet18-p1", "layer3"),
        ("cifar100-resnet18-p1", "layer4"),
    ],
)
def test_approximator_homogenous_low_rank_have_same_inits_(model, layer):
    model = models.get_model(model)

    approx1 = approximators.construct_approximator_for(
        model=model,
        layer=layer,
        compression_ratio=1.0,
        mode=approximators.ApproximatorMode.HOMOGENOUS_LOWRANK,
        seed=1,
    )

    approx2 = approximators.construct_approximator_for(
        model=model,
        layer=layer,
        compression_ratio=1.0,
        mode=approximators.ApproximatorMode.HOMOGENOUS_LOWRANK,
        seed=1,
    )

    approx3 = approximators.construct_approximator_for(
        model=model,
        layer=layer,
        compression_ratio=1.0,
        mode=approximators.ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER,
        seed=1,
    )

    assert count_params_in_model(approx1) == count_params_in_model(approx2)

    for param1, param2 in zip(approx1.parameters(), approx2.parameters()):
        np.testing.assert_allclose(param1, param2)

    approx1_backbone = approx1[0]
    approx3_backbone = approx3[0]
    for param1, param2 in zip(
        approx1_backbone.parameters(), approx3_backbone.parameters()
    ):
        np.testing.assert_allclose(param1, param2)
