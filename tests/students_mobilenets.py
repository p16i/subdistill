import numpy as np
import pytest
import torch

from xaikd import models, utils


@pytest.mark.parametrize(
    "student_name,class_indices",
    [
        ("student-mobilenets", np.arange(10)),
        ("student-mobilenetl", np.arange(10)),
    ],
)
@pytest.mark.parametrize(
    "layer",
    [
        "features.8",
        "features.12",
    ],
)
def test_student_callable(student_name, class_indices, layer):
    trng = torch.Generator()
    trng.manual_seed(1)
    bs = 7
    nc = len(class_indices)
    inp = torch.rand((bs, 3, 224, 224), generator=trng)
    model = models.get_untrained_model(
        student_name, num_classes=nc, class_indices=class_indices
    )

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


@pytest.mark.skip(reason="obsolete")
@torch.no_grad()
def test_get_trained_student_has_correct_layer_layer():
    class_indices = [10, 20, 50, 20]
    nc = len(class_indices)
    student = models.get_untrained_model(
        "student-mobilenets-trained", num_classes=nc, class_indices=class_indices
    )
    original = models.get_trained_model("imagenet-mobilenets-tv")

    expected_weight = original.classifier[-1].weight[class_indices].numpy()
    actual_weight = student.classifier[-1].weight.numpy()
    np.testing.assert_allclose(actual_weight, expected_weight)

    expected_bias = original.classifier[-1].bias[class_indices].numpy()
    actual_bias = student.classifier[-1].bias.numpy()
    np.testing.assert_allclose(actual_bias, expected_bias)


@pytest.mark.skip(reason="obsolete")
@pytest.mark.parametrize(
    "arch",
    [
        "student-mobilenetxs-cifar",
        "student-mobilenetxs-cifarv2",
        "student-mobilenetxxs-cifar",
        "student-mobilenetxxs-cifarv2",
    ],
)
def test_student_callable_cifar(arch):
    x = torch.randn(2, 3, 32, 32)

    student = models.get_untrained_model(arch, num_classes=10)

    assert np.isfinite(student(x).detach().cpu().numpy()).all()
