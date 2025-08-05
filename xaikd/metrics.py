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
        dataloader,
        desc="computing accuracy for selected claseses",
        disable=not verbose,
        miniters=10,
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

        raise NotImplementedError("obsolete this and use AUROCBinXent")
        assert not model.training

        metric_auroc = BinaryAUROC(thresholds=100)

        for x, y in tqdm(dataloader, desc="Computing AUROC", disable=not verbose):
            logodd = model(x.to(device)).cpu()

            assert np.isin(y.numpy(), [0, 1]).all()

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


class MetricRecon(MetricFunction):
    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ) -> typing.Tuple[float]:
        assert not model.training

        metric = MeanMetric()
        for batch in tqdm(
            dataloader,
            desc=f"[evaluating reconstruction error",
            disable=not verbose,
        ):
            x = batch[0]

            x = x.to(device)

            output = model(x).detach().cpu()  # Ensure logits are on CPU

            norm = torch.linalg.norm(output, ord=2, dim=1)

            for kix, k in enumerate(arr_ks):
                Uk = U[:, :k]
                hook = None
                try:
                    module = intercepts.get_module_for_layer(model=model, layer=layer)
                    hook = module.register_forward_hook(
                        intercepts.construct_fh_with_projection(
                            Uk,
                            shape_normalizer=feature_map_shape_normalizer,
                            device=device,
                        )
                    )
                    recon_output = model(x).detach().cpu()

                    np.testing.assert_equal(len(output), len(recon_output))

                    err = torch.linalg.norm(
                        output - recon_output, ord=2, dim=1
                    )  # Compute reconstruction error
                    arr_metric_recon.update(kix, err.cpu())

                    # fixme add cosine
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        output, recon_output, dim=1
                    )
                    arr_metric_cosine.update(kix, cosine_sim.cpu())
                finally:
                    if hook is not None:
                        hook.remove()

    def _metric_names(self) -> typing.Tuple[str]:
        return ["recon"]


class MetricAUROCBinaryCrossEntropy(MetricFunction):
    def __init__(self, convert_auroc=True):
        self.convert_auroc = convert_auroc

    @torch.no_grad()
    def __call__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
        verbose=False,
        prefix=None,
    ):

        assert not model.training

        metric_auroc = BinaryAUROC(thresholds=100)
        metric_mean = MeanMetric()

        desc = "Computing AUROC"
        if prefix is not None:
            desc = f"[{prefix}] {desc}"

        for x, y in tqdm(dataloader, desc=desc, disable=not verbose, miniters=10):
            n = x.shape[0]
            logodd = model(x.to(device)).cpu()

            assert torch.isfinite(logodd).all()

            if len(logodd.shape) == 2:
                assert logodd.shape == (n, 1)
                logodd = logodd.squeeze(1)

            assert len(logodd.shape) == 1, f"{logodd.shape}"

            assert np.isin(y.numpy(), [0, 1]).all()

            metric_auroc.update(logodd, y)
            loss = F.binary_cross_entropy_with_logits(
                logodd, y.float(), reduction="none"
            )
            assert len(loss.shape) == 1
            assert logodd.shape == (n,)
            assert loss.shape == (n,)
            metric_mean.update(loss)

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
        for x, y in tqdm(
            dataloader, desc="Computing ACC", disable=not verbose, miniters=10
        ):
            logits = model(x.to(device)).cpu()

            assert len(logits.shape) == 2, f"{logits.shape}"
            assert logits.shape[1] == self.num_classes

            metric.update(logits, y)

        metric = float(metric.compute())

        return (metric,)

    def _metric_names(self):
        return ("accuracy",)


class MetricAccuracyXent(MetricFunction):
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    @torch.no_grad()
    def __call__(
        self, model: nn.Module, dataloader: DataLoader, device: str, verbose=False
    ):

        assert not model.training

        metric_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        metric_xent = MeanMetric()
        for x, y in tqdm(
            dataloader, desc="Computing ACC", disable=not verbose, miniters=10
        ):
            logits = model(x.to(device)).cpu()

            assert len(logits.shape) == 2, f"{logits.shape}"
            assert logits.shape[1] == self.num_classes

            xent = F.cross_entropy(logits, y)

            metric_acc.update(logits, y)
            metric_xent.update(xent)

        metric_acc = float(metric_acc.compute())
        metric_xent = float(metric_xent.compute())

        return (metric_acc, metric_xent)

    def _metric_names(self):
        return ("accuracy", "xent")
