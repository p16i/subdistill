import numpy as np
import pytest
import torch

from xaikd import models, utils


@pytest.mark.parametrize(
    "student_name",
    [
        "student-mobilenetv4-small",
        "student-mobilenetv4-small-alternative-init",
    ],
)
@pytest.mark.parametrize(
    "layer",
    [
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    ],
)
def test_student_v4_callable(student_name, layer):
    trng = torch.Generator()
    trng.manual_seed(1)
    bs = 7
    nc = 10
    inp = torch.rand((bs, 3, 224, 224), generator=trng)
    model = models.get_untrained_model(student_name, num_classes=nc)

    assert model.training

    output = model(inp)
    output.sum().backward()

    assert output.shape == (bs, nc)

    total, trainable = utils.count_params_in_model(model)

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=False, detach_output=False
        )

        output = model(inp)

        assert output.shape[1] == nc

        utils.interceptor.get_output(module)

    finally:
        hook.remove()
