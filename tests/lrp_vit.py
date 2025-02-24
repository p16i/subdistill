import pytest

import numpy as np
import torch
from torchvision import transforms as T

from zennit.attribution import Gradient
from xaikd import models, attributors, lrp, logit_modifiers


from PIL import Image


@pytest.mark.slow
def test_callable():
    torch.manual_seed
    rng = torch.Generator().manual_seed(1)
    vit = models.get_trained_model("imagenet-vitb-tv")

    lb = torch.zeros(3).reshape((1, -1, 1, 1))
    hb = torch.ones(3).reshape((1, -1, 1, 1))

    input = torch.randn(1, 3, 224, 224, generator=rng)
    with torch.no_grad():
        expected_output = vit(input).numpy()

    assert hasattr(vit.encoder, "layers")

    with Gradient(
        model=vit, composite=lrp.vit._build_composite(lb=lb, hb=hb)
    ) as attributor:
        pass
        actual_output, hm = attributor.forward(input, lambda l: l)

        assert np.isfinite(actual_output.detach().numpy()).any()
        assert np.isfinite(hm.detach().numpy()).any()

        actual_output = actual_output.detach().numpy()

    assert hasattr(vit.encoder, "layers")

    for helper_attribute in [
        "_ln",
        "_layers",
        "_dropout",
    ]:
        assert not hasattr(vit.encoder, helper_attribute)

    np.testing.assert_allclose(actual_output, expected_output, atol=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize(
    "filename,target",
    [
        ("./tests/data/castle.jpg", 483),
        ("./tests/data/viaduct.jpg", 888),
        ("./tests/data/volcano.jpg", 980),
        ("./tests/data/zebra.jpg", 340),
    ],
)
def test_callable_and_no_nan(filename, target):
    model = models.get_trained_model("imagenet-vitb-tv")

    lb = torch.zeros(3).reshape((1, -1, 1, 1))
    hb = torch.ones(3).reshape((1, -1, 1, 1))

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalizer])

    input = transform(Image.open(filename)).unsqueeze(0)
    target = torch.tensor([target])

    logit_mod = logit_modifiers.MultiClassTargetLogit()

    with Gradient(
        model=model, composite=attributors.get_arch_specific_composite(model, lb, hb)
    ) as attributor:
        output, hm = attributor(input, lambda logits: logit_mod(logits, target))

        assert np.isfinite(output.detach().numpy()).any()
        assert np.isfinite(hm.detach().numpy()).any()
