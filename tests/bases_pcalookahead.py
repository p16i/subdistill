import pytest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from collections import OrderedDict


from xaikd import datasets, bases, models, attributors
from xaikd import utils

from torch.utils.data import random_split


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

    ds, _ = random_split(dataset.create_subset(train_split=False), [0.1, 0.9])

    dataloader = datasets.build_dataloader(ds, shuffle=False)

    logit_modifier = attributors.WinningClassEvidence(len(dataset.selected_classes))

    arr_act, _ = attributors.extract_activation_context(
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
        arr_ctx=None,
        **dict(model=model, layer=layer, dataloader=dataloader)
    )

    pcaah.construct_adapter(k=d - 2, mode=bases.AdapterMode.ENCODER, device="cpu")

    assert True
