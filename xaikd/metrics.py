import typing
import numpy as np

from abc import ABC, abstractmethod, ABCMeta


from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, MeanMetric
from torchmetrics.classification import BinaryAUROC

from tqdm import tqdm


def accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: str,
    verbose=False,
) -> typing.Tuple[float, float]:
    """_summary_

    Args:
        model (nn.Module): _description_
        dataloader (DataLoader): _description_
        num_classes (int): _description_
        device (str): _description_
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        acc: torch.Tensor
        xent: torch.Tensor
    """
    model.eval()

    metric_acc = Accuracy(task="multiclass", num_classes=num_classes)
    metric_xent = MeanMetric()

    for x, y in tqdm(
        dataloader, desc="computing accuracy for selected claseses", disable=not verbose
    ):
        logits = model(x.to(device)).cpu()
        metric_acc.update(logits, y)
        xent = F.cross_entropy(logits, y, reduction="none")
        metric_xent.update(xent)

    return float(metric_acc.compute()), float(metric_xent.compute())


class MetricFunction(ABC):
    @abstractmethod
    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ) -> typing.Tuple[float, ...]:
        pass

    def __str__(self) -> str:
        return "-".join(self._metric_names())

    @abstractmethod
    def _metric_names(self) -> typing.Tuple[str, ...]:
        pass


class MetricAUROC(MetricFunction):
    def __init__(self, convert_auroc=True):
        self.convert_auroc = convert_auroc

    @torch.no_grad()
    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ):

        assert not model.training

        metric_auroc = BinaryAUROC(thresholds=100)

        for x, y in tqdm(dataloader, desc="Computing AUROC", disable=not verbose):
            logodd = model(x.to(device)).cpu()

            assert len(logodd.shape) == 1, f"{logodd.shape}"

            metric_auroc.update(logodd, y)

        auroc = metric_auroc.compute()
        if self.convert_auroc:
            auroc = np.max([auroc, 1 - auroc])

            assert 0.5 <= auroc <= 1.0

        auroc = float(auroc)

        return (auroc,)

    def _metric_names(self):
        return ("auroc",)


class MetricAUROCBinaryCrossEntropy(MetricFunction):
    def __init__(self, convert_auroc=True):
        self.convert_auroc = convert_auroc

    @torch.no_grad()
    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ):

        assert not model.training

        metric_auroc = BinaryAUROC(thresholds=100)
        metric_mean = MeanMetric()

        for x, y in tqdm(dataloader, desc="Computing AUROC", disable=not verbose):
            logodd = model(x.to(device)).cpu()

            assert len(logodd.shape) == 1, f"{logodd.shape}"

            metric_auroc.update(logodd, y)
            metric_mean.update(F.binary_cross_entropy_with_logits(logodd, y))

        auroc = metric_auroc.compute()
        if self.convert_auroc:
            auroc = np.max([auroc, 1 - auroc])

            assert 0.5 <= auroc <= 1.0

        auroc = float(auroc)
        binxent = float(metric_mean.compute())

        return (auroc, binxent)

    def _metric_names(self):
        return ("auroc", "binxent")


class MetricAccuracy(MetricFunction):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ):

        assert not model.training

        metric = Accuracy(task="multiclass", num_classes=self.num_classes)
        for x, y in tqdm(dataloader, desc="Computing ACC", disable=not verbose):
            logits = model(x.to(device)).cpu()

            assert len(logits.shape) == 2, f"{logits.shape}"
            assert logits.shape[1] == self.num_classes

            metric.update(logits, y)

        metric = float(metric.compute())

        return (metric,)

    def _metric_names(self):
        return ("accuracy",)
