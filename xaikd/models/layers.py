import typing

import torch

from torch import nn

from xaikd import datasets


def resolve_teacher_last_layer(dataset: datasets.DatasetConfiguration) -> nn.Module:

    if isinstance(dataset, datasets.celeba.CelebAAttribute):
        return TaskLogitSelection(task_id=dataset.attr_ix)
    elif isinstance(
        dataset, datasets.cifar100.some_vs_others.CIFAR100SuperclassVsOthers
    ):
        return LayerLogOddSelectedClasses(selected_classes=dataset.selected_classes)
    else:
        raise


# todo: add test
class LayerLogOddSelectedClasses(nn.Module):
    def __init__(self, selected_classes: typing.List[int]) -> None:
        super().__init__()

        self.selected_classes = selected_classes

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        _, total_classes = logits.shape

        other_classes = list(
            set(range(total_classes)).difference(self.selected_classes)
        )

        arr_pos_logits = logits[:, self.selected_classes]
        arr_neg_logits = logits[:, other_classes]

        logodd = torch.logsumexp(arr_pos_logits, dim=1) - torch.logsumexp(
            arr_neg_logits, dim=1
        )

        assert torch.isfinite(logodd).all()

        return logodd


# todo: add test
class SubclassSelection(nn.Module):
    def __init__(self, selected_classes: typing.List[int]) -> None:
        super().__init__()

        self.selected_classes = selected_classes

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, self.selected_classes]


class TaskLogitSelection(nn.Module):
    def __init__(self, task_id) -> None:
        super().__init__()
        self.task_id = task_id

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        assert isinstance(logits, torch.Tensor)

        _, num_tasks = logits.shape

        return logits[:, self.task_id]
