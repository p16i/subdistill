import numpy as np
import pytest
import torch

from xaikd import constants, models, utils


@pytest.mark.parametrize("prefix,num_layers", [("vitstudent", 4), ("vitstudent6l", 6)])
@pytest.mark.parametrize("hidden_dim", constants.ARR_VIT_STUDENT_HIDDEN_DIMENSIONS)
@pytest.mark.parametrize(
    "layer",
    [
        "encoder.layers.0",
    ],
)
def test_student_callable(prefix, num_layers, hidden_dim, layer):
    trng = torch.Generator()
    trng.manual_seed(1)
    bs = 7
    nc = 10
    inp = torch.rand((bs, 3, 224, 224), generator=trng)
    model = models.get_untrained_model(f"{prefix}-{hidden_dim}", num_classes=nc)

    assert model.training

    output = model(inp)
    output.sum().backward()

    assert output.shape == (bs, nc)

    assert len(model.encoder.layers) == num_layers

    total, trainable = utils.count_params_in_model(model)

    print(
        f"num parametesr (hidden_dims={hidden_dim}): total={total}, trainable={trainable}",
    )

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=False, detach_output=False
        )

        _ = model(inp)

        act = utils.interceptor.get_output(module)

        assert len(act.shape) == 4
        assert act.shape == (bs, hidden_dim, 197, 1)
    finally:
        hook.remove()


# @torch.no_grad()
# def test_canonize_student():
#     torch.manual_seed(1)
#     x_train = torch.rand(5, 3, 224, 224)
#     num_classes = 6

#     model = models.get_untrained_model("student-32-24-16-8", num_classes=num_classes)
#     model(x_train)

#     model.eval()

#     canonized_model = canonize_student_model(model)

#     canonized_model.eval()

#     x = torch.rand(5, 3, 224, 224)

#     expected = model(x)
#     actual = canonized_model(x)

#     np.testing.assert_allclose(actual, expected, atol=1e-6)
