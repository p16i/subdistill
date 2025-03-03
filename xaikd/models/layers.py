import typing

import torch

from torch import nn


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
