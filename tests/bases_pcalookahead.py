import pytest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict


from xaikd import bases, models, attributors
from xaikd import utils

from torch.utils.data import random_split

from xaikd import datasets

pytest.skip("obsolete", allow_module_level=True)


@pytest.mark.slow
@pytest.mark.gpu
def test_pcalookahead_trainable():
    rng = np.random.default_rng(seed=1)
    dout = 4
    torch.manual_seed(1)

    device = utils.get_device()
    din = 3
    d = 5

    layer = "layer4"

    dataset = datasets.construct("cifar100-people")
    model = models.get_trained_model("cifar100-resnet18-v1")
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)
    model.to(device)

    ds, _ = random_split(dataset.create_subset(train_split=False), [0.1, 0.9])

    dataloader = datasets.build_dataloader(ds, shuffle=False)

    logit_modifier = attributors.WinningClassEvidence(len(dataset.selected_classes))

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
        data_loader=dataloader,
        logit_modifier=logit_modifier,
        rng=rng,
        device=device,
    )

    pcaah = bases.get_basis("pcalookahead--uncentered")
    pcaah.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        **dict(model=model, layer=layer, dataloader=dataloader),
    )

    pcaah.construct_adapter(k=d - 2, mode=bases.AdapterMode.ENCODER, device=device)

    assert True


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize(
    "teacher,layer,ref_basis_name",
    [
        ("imagenet-resnet18-tv", "layer3", "prca-sortabs--uncentered"),
        ("imagenet-vitb-tv", "encoder.layers.5", "pca--uncentered"),
    ],
)
def test_initialization_conditions(teacher, layer, ref_basis_name):
    rng = np.random.default_rng(seed=1)

    device = utils.get_device()

    dataset = datasets.construct("imagenet-butterfly")
    model = models.get_trained_model(teacher)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)
    model.to(device)

    ds, _ = random_split(dataset.create_subset(train_split=False), [0.1, 0.9])

    dataloader = datasets.build_dataloader(ds, shuffle=False)

    logit_modifier = attributors.WinningClassEvidence(len(dataset.selected_classes))

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        layer=layer,
        dataset=dataset,
        data_loader=dataloader,
        logit_modifier=logit_modifier,
        rng=rng,
        device=device,
    )

    pcaah = bases.get_basis(f"pcalookahead--uncentered")
    pcaah.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        **dict(model=model, layer=layer, dataloader=dataloader),
    )

    ref_basis = bases.get_basis(f"{ref_basis_name}")
    ref_basis.fit(arr_act=arr_act, arr_ctx=arr_ctx)

    np.testing.assert_allclose(pcaah.U, ref_basis.U)
