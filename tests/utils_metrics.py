import pytest
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from xaikd.utils import metrics


@pytest.mark.skip
def test_accuracy_with_subclasses():
    considered_classes = [1, 2, 3]
    model = nn.Identity()
    x = torch.Tensor(
        [
            [8, 0, 5, 3, 2, 1],
            [2, 3, 0, 2, 4, 1],
            [2, 0, 0, 5, 10, 1],
            # the entry below is assumed to be false (target is 3).
            [1, 5, 0, 3, 10, 1],
        ]
    )
    y = torch.Tensor([2, 1, 3, 3])

    ds = TensorDataset(x, y)

    dl = DataLoader(ds, batch_size=3, shuffle=True)

    def transform_target(target: torch.Tensor) -> torch.Tensor:
        target_transform_dict = dict(
            zip(considered_classes, range(len(considered_classes)))
        )
        new_target = []

        for t in target:
            new_target.append(target_transform_dict[int(t.detach().cpu())])

        return torch.Tensor(new_target).long().to(target.device)

    acc, xent = metrics.accuracy_with_subclasses(
        model,
        dl,
        considered_classes=considered_classes,
        transform_target=transform_target,
        device="cpu",
    )
    np.testing.assert_allclose(acc, 0.75)
    np.testing.assert_allclose(
        xent, F.cross_entropy(x[:, considered_classes], transform_target(y))
    )

    acc2, xent2 = metrics.accuracy_with_subclasses(
        model,
        dl,
        considered_classes=considered_classes,
        transform_target=transform_target,
        device="cpu",
    )

    np.testing.assert_allclose(acc2, acc, err_msg="shuffle should NOT affect metric!")
    np.testing.assert_allclose(xent2, xent, err_msg="shuffle should NOT affect metric!")


@pytest.mark.parametrize(
    "order,expected", [([0, 2, 1], [0.01, 0.04, 0]), ([0, 1, 2], [0.01, 0.01, 0.0])]
)
def test_unexplained_relevance(order, expected):
    activation = np.array([[1, -1, 1]])
    context = np.array([[1, 0.2, 0.1]])

    U = np.eye(3)[:, order]

    stats = metrics.unexplained_relevance(activation, context, U)

    total = (activation * context).sum()
    np.testing.assert_allclose(stats, [total**2] + expected, atol=1e-6)


# todo: all metrics should not change batch norm stats
