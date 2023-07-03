import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from xaikd.utils import metrics


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

    dl = DataLoader(ds)

    def transform_target(target: torch.Tensor) -> torch.Tensor:
        target_transform_dict = dict(
            zip(considered_classes, range(len(considered_classes)))
        )
        new_target = []

        for t in target:
            new_target.append(target_transform_dict[int(t.detach().cpu())])

        return torch.Tensor(new_target).to(target.device)

    acc = metrics.accuracy_with_subclasses(
        model,
        dl,
        considered_classes=considered_classes,
        transform_target=transform_target,
        device="cpu",
    )

    assert acc == 0.75
