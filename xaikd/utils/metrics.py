import typing
import numpy as np

import torch

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy

from xaikd import attributors, bases


def auroc(
    model: nn.Module,
    dataloader: DataLoader,
    classes: typing.Tuple[int, int],
    device: str,
    should_convert_auroc=False,
) -> typing.Tuple[float, float]:
    model.eval()
    c1, c2 = classes

    metric_auroc = BinaryAUROC()
    metric_binary = BinaryAccuracy()
    model = model.to(device)
    for x, y in dataloader:
        logits = model(x.to(device)).cpu()

        logodd = logits[:, c1] - logits[:, c2]

        assert np.logical_or(y == c1, y == c2).all()

        ybin = torch.from_numpy(np.where(y == c1, 0, 1))
        metric_auroc.update(logodd, ybin)
        metric_binary.update((logodd < 0).int(), ybin.int())

    auroc = metric_auroc.compute()
    if should_convert_auroc:
        auroc = np.max([auroc, 1 - auroc])

    bin_accuracy = metric_binary.compute()

    return float(auroc), float(bin_accuracy)


def accuracy(model: nn.Module, dl: DataLoader, num_classes: int, device: str) -> float:
    model.eval()
    metric = Accuracy(task="multiclass", num_classes=num_classes)

    for x, y in dl:
        logits = model(x.to(device)).cpu()
        metric.update(logits, y)

    return float(metric.compute())


def accuracy_with_subclasses(
    model: nn.Module,
    dl: DataLoader,
    considered_classes: typing.List[int],
    transform_target: typing.Callable[[torch.Tensor], torch.Tensor],
    device: str,
) -> float:
    model.eval()

    metric = Accuracy(task="multiclass", num_classes=len(considered_classes))

    for x, y in dl:
        logits = model(x.to(device)).cpu()
        selected_logits = logits[:, considered_classes]
        transformed_y = transform_target(y)
        metric.update(selected_logits, transformed_y)

    return float(metric.compute())


def auroc_with_basis(
    model: nn.Module,
    module: nn.Module,
    dataloader: DataLoader,
    classes: typing.Tuple[int, int],
    basis: bases.Basis,
    device: str,
    arr_ks: typing.List[int],
    should_convert_auroc: bool,
) -> typing.List[float]:
    model.eval()

    arr_aurocs = []

    for k in tqdm(arr_ks, desc=f"[basis={basis}]"):
        try:
            hook = module.register_forward_hook(
                basis.construct_fh_rank_k_projection(k, device=device)
            )

            value = auroc(
                model,
                dataloader=dataloader,
                classes=classes,
                device=device,
                should_convert_auroc=should_convert_auroc,
            )

            arr_aurocs.append(value)

        finally:
            hook.remove()

    return arr_aurocs
