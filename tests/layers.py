import pytest
import numpy as np

import torch

from xaikd.models import layers


@pytest.mark.parametrize("task_id", [0, 1, 2])
def test_task_logit_selection(task_id):
    layer = layers.TaskLogitSelection(task_id=task_id)

    logits = torch.randn(10, 1000)

    actual = layer(logits).numpy()
    expected = logits[:, task_id].numpy()

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "selected_classes",
    [
        (0, 1, 2),
        (74, 5, 9),
    ],
)
def test_logodd_selected_classes(selected_classes):
    other_classes = set(range(1000)).difference(selected_classes)

    layer = layers.LayerLogOddSelectedClasses(selected_classes=selected_classes)

    logits = torch.randn(10, 1000)

    actual = layer(logits).numpy()

    expected = torch.logsumexp(logits[:, selected_classes], dim=1) - torch.logsumexp(
        logits[:, list(other_classes)], dim=1
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "selected_classes",
    [
        (0, 1, 2),
        (74, 5, 9),
    ],
)
def test_subsclasses_classes(selected_classes):

    layer = layers.SubclassSelection(selected_classes=selected_classes)

    logits = torch.randn(10, 1000)

    actual = layer(logits).numpy()

    expected = logits[:, selected_classes]

    np.testing.assert_allclose(actual, expected)
