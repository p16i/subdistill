import numpy as np
import pytest
import torch

import timm

from xaikd import constants, models, utils


@pytest.mark.slow()
@pytest.mark.parametrize(
    "layer,hidden_dim,feature_map_size",
    [
        ("stages.0", 32, 56),
        ("stages.1", 48, 28),
        ("stages.2", 96, 14),
        ("stages.3", 176, 7),
    ],
)
def test_student_callable_and_have_correct_featuremap_shape(
    layer, hidden_dim, feature_map_size
):
    model_name = "student-efficientformerv2_s0"
    trng = torch.Generator()
    trng.manual_seed(1)
    bs = 7
    nc = 10
    inp = torch.rand((bs, 3, 224, 224), generator=trng)

    model: timm.models.EfficientFormerV2 = models.get_untrained_model(model_name, num_classes=nc)  # type: ignore

    assert model.training

    output = model(inp)
    output.sum().backward()

    assert output.shape == (bs, nc)

    np.testing.assert_equal(len(model.stages), 4)

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=False, detach_output=False
        )

        _ = model(inp)

        act = utils.interceptor.get_output(module)

        assert len(act.shape) == 4
        np.testing.assert_equal(
            act.shape, (bs, hidden_dim, feature_map_size, feature_map_size)
        )
    finally:
        hook.remove()
