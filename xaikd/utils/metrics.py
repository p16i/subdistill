import numpy as np

import torch

from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.classification import BinaryAUROC

from xaikd import attributors


def estimate_auroc(
    model: nn.Module,
    dataloader: DataLoader,
    logodd_mod: attributors.LogOddEvidence,
    device: str,
) -> float:
    class1, class2 = logodd_mod.classes

    metric = BinaryAUROC()
    for x, y in dataloader:
        logits = model(x.to(device))

        logodd = logodd_mod(logits).sum(dim=1).detach().cpu()

        assert np.logical_or(y == class1, y == class2).all()

        ybin = np.where(y == class1, 0, 1)
        metric.update(logodd, torch.from_numpy(ybin))

    auroc = metric.compute()

    return float(auroc)
