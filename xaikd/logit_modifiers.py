from abc import ABC, abstractmethod
import typing


import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


LOGIT_MODIFIERS = dict()


def register(fn):
    def wrapper(cls):
        slug = cls.__name__
        assert not (slug in LOGIT_MODIFIERS)
        LOGIT_MODIFIERS[slug] = cls
        return cls


class LogitModifier(ABC):
    @abstractmethod
    def __call__(
        self, logits: torch.Tensor, targets: typing.Union[torch.Tensor, None]
    ) -> torch.Tensor:
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class MultiClassTargetLogit(LogitModifier):
    def __call__(self, logits, targets) -> torch.Tensor:
        _, num_classes = logits.shape
        logits = logits.clone()
        return logits * F.one_hot(targets, num_classes).to(logits.device)


class MultiClassAllLogits(LogitModifier):
    def __call__(self, logits, targets) -> torch.Tensor:
        logits = logits.clone()
        return logits


class MultiClassWinningLogit(LogitModifier):
    def __call__(self, logits, targets) -> torch.Tensor:
        _, num_classes = logits.shape
        logits = logits.clone()
        wining_targets = torch.argmax(logits, dim=1)
        return logits * F.one_hot(wining_targets, num_classes).to(logits.device)


class MultiClassZeroLogit(LogitModifier):
    def __call__(self, logits, targets) -> torch.Tensor:
        logits = logits.clone()
        return torch.zeros_like(logits).to(logits.device)


class MultiClassDifferenceTop2Logits(LogitModifier):
    def __call__(self, logits, targets) -> torch.Tensor:
        _, num_classes = logits.shape
        logits = logits.clone()
        # find the label of two winning classes
        _, indices = torch.topk(logits, dim=1, k=2)
        return logits * F.one_hot(indices[:, 0], num_classes) - logits * F.one_hot(
            indices[:, 1], num_classes
        )


class MultiClassLogOddWinning(LogitModifier):
    def __call__(self, logits: torch.Tensor, targets=None) -> torch.Tensor:
        (n, d) = logits.shape

        values, indices = torch.topk(logits, dim=1, k=d)
        assert values.shape == logits.shape

        logit_winning = values[:, 0]
        logit_others = values[:, 1:]
        assert logit_others.shape == (n, d - 1)
        logodd = logit_winning - torch.logsumexp(logit_others, dim=1)

        assert torch.isfinite(logodd).all()

        assert logodd.shape == (n,)

        return logodd


class BinaryLogOddWinning(LogitModifier):
    def __init__(
        self,
        threshold: float,
    ) -> None:
        self.threshold = threshold

    def __call__(self, logits: torch.Tensor, targets=None) -> torch.Tensor:
        (n,) = logits.shape

        assert len(logits.shape) == 1

        return torch.abs(logits - self.threshold)
