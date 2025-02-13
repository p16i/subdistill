import typing
import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod

import torch

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, MeanMetric
from torchmetrics.classification import BinaryAUROC

from tqdm import tqdm


def auroc(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    should_convert_auroc=True,
) -> float:
    raise NotImplementedError("obsolete")


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


def accuracy_with_subclasses(
    model: nn.Module,
    dataloader: DataLoader,
    considered_classes: typing.List[int],
    transform_target: typing.Callable[[typing.List[int]], typing.List[int]],
    device: str,
    verbose=False,
) -> typing.Tuple[float, float]:
    raise NotImplementedError("Obsolete! Use accuracy(..) instead!")
    model.eval()

    metric_acc = Accuracy(task="multiclass", num_classes=len(considered_classes))
    metric_xent = MeanMetric()

    for x, y in tqdm(
        dataloader, desc="Computing accuracy for selected claseses", disable=not verbose
    ):
        logits = model(x.to(device)).cpu()
        selected_logits = logits[:, considered_classes]
        transformed_y = transform_target(y.detach().cpu().numpy())
        transformed_y = torch.Tensor(transformed_y).to(y.device)
        metric_acc.update(selected_logits, transformed_y)
        xent = F.cross_entropy(selected_logits, transformed_y, reduction="none")
        metric_xent.update(xent)

    return float(metric_acc.compute()), float(metric_xent.compute())


# def auroc_with_basis(
#     model: nn.Module,
#     module: nn.Module,
#     dataloader: DataLoader,
#     classes: typing.Tuple[int, int],
#     basis: bases.OrthogonalBasis,
#     device: str,
#     arr_ks: typing.List[int],
#     should_convert_auroc: bool,
# ) -> typing.List[float]:
#     raise NotImplementedError("obsolete")
#     model.eval()

#     arr_aurocs = []

#     for k in tqdm(arr_ks, desc=f"[basis={basis}]"):
#         try:
#             hook = module.register_forward_hook(
#                 basis.construct_fh_rank_k_projection(k, device=device)
#             )

#             value = auroc(
#                 model,
#                 dataloader=dataloader,
#                 classes=classes,
#                 device=device,
#                 should_convert_auroc=should_convert_auroc,
#             )

#             arr_aurocs.append(value)

#         finally:
#             hook.remove()

#     return arr_aurocs


def unexplained_relevance(
    activation: npt.NDArray, context: npt.NDArray, U: npt.NDArray
) -> typing.List[float]:
    n, d = activation.shape
    total_relevance = (activation * context).sum(axis=1)
    assert total_relevance.shape == (n,)

    relevance_per_dim = (activation @ U) * (context @ U)

    assert relevance_per_dim.shape == (n, d)

    stats = [float(np.mean(total_relevance**2))]

    for k in range(1, d + 1):
        explained_relevance_sofar = relevance_per_dim[:, :k]
        np.testing.assert_equal(explained_relevance_sofar.shape, (n, k))
        unexplained_relevance = (
            total_relevance - explained_relevance_sofar.sum(axis=1)
        ) ** 2

        assert unexplained_relevance.shape == (n,)

        stats.append(float(np.mean(unexplained_relevance)))

    return stats


class MetricFunction(ABC):
    @abstractmethod
    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ) -> typing.Dict[str, float]:
        pass


class MetricAUROC(MetricFunction):
    def __init__(self, convert_auroc=True):
        self.convert_auroc = convert_auroc

    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ) -> typing.Dict[str, float]:

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

        return dict(auroc=float(auroc))
