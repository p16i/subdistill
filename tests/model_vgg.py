import pytest

import torch
from xaikd import models, utils
from xaikd.models.vgg import VGGFeatureBlocks

from torchvision import models as tvm
import numpy as np


@torch.no_grad()
@pytest.mark.parametrize(
    "arch_cls,num_classes,input_size",
    [
        (tvm.vgg11, 100, (10, 3, 32, 32)),
        (tvm.vgg16, 1000, (10, 3, 224, 224)),
    ],
)
def test_group_feature_layers_vgg11(arch_cls, num_classes, input_size):
    model = arch_cls(num_classes=num_classes)
    model.eval()

    models_with_block = VGGFeatureBlocks(model)

    x = torch.randn(input_size)

    np.testing.assert_equal(
        utils.count_params_in_model(model),
        utils.count_params_in_model(models_with_block),
    )
    np.testing.assert_allclose(model(x), models_with_block(x))


@pytest.mark.skip("[todo]")
def test_split_vgg11_model(slug, layer):
    pass
