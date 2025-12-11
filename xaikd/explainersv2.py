from abc import ABC, abstractmethod
from typing import Tuple
import numpy.typing as npt


from functools import partial

import torch
from torch import nn

from torch.nn import functional as F


from zennit.attribution import Gradient  # type: ignore

import numpy as np
from numpy import typing as npt

from captum.attr import IntegratedGradients, ShapleyValueSampling  # type: ignore
from zennit.torchvision import ResNetCanonizer  # type: ignore
from zennit.composites import EpsilonGammaBox  # type: ignore
from zennit.attribution import Gradient  # type: ignore

from torchvision import transforms as T  # type: ignore

EXPLAINERS = dict()

NORMALIZER = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class LogitGapWrtTarget(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits):
        output = torch.zeros_like(logits)
        for cix in range(self.num_classes):
            others = set(range(self.num_classes)) - {cix}

            output[:, cix] = logits[:, cix] - logits[:, list(others)].max(dim=1).values

        return output


def logit_gap_wrt_target(logits, target, num_classes):
    device = logits.device
    target_logit = logits * F.one_hot(target, num_classes=num_classes).float().to(
        device
    )

    with torch.no_grad():
        mod_logit = logits - 1e6 * target_logit
        _, indices = torch.topk(
            mod_logit, dim=1, k=2  # this make sure that target class is not selected
        )

    other_logits = logits * F.one_hot(
        indices[:, 0], num_classes=num_classes
    ).float().to(device)
    out = target_logit - other_logits

    return out


def register_explainer(name):
    def wrapped(fn):
        EXPLAINERS[name] = fn

        return fn

    return wrapped


class Explainer(ABC):
    def __init__(self, model: nn.Module, num_classes: int, device: str) -> None:
        self.model = model
        self.num_classes = num_classes
        self.device = device

    @abstractmethod
    def attribute(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        pass


@register_explainer("lrpresnet")
class LRPResNetExplainer(Explainer):
    def __init__(self, model: nn.Module, num_classes: int, device: str):
        super().__init__(model, num_classes, device)

        self.gamma = 1

        self.low, self.high = NORMALIZER(
            torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]])
        )

        # create a composite, specifying the canonizers, if any
        self.composite = EpsilonGammaBox(
            low=self.low,
            high=self.high,
            canonizers=[ResNetCanonizer()],
            gamma=self.gamma,
        )

    def attribute(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        nb = x.shape[0]
        # create the attributor, specifying model and composite
        with Gradient(model=self.model, composite=self.composite) as attributor:
            # compute the model output and attribution
            x = x.to(self.device)

            logit, attribution = attributor(
                x, lambda logits: logit_gap_wrt_target(logits, y, self.num_classes)
            )  # type: ignore

        assert logit.shape == (nb, self.num_classes)
        logit = logit.detach().cpu().numpy()

        attribution = attribution.sum(dim=1).detach().cpu().numpy()
        assert attribution.shape == (nb, 224, 224), attribution.shape

        return logit, attribution


@register_explainer("intgrad")
class IntegratedGradientsExplainer(Explainer):
    def __init__(self, model: nn.Module, num_classes: int, device: str) -> None:
        super().__init__(model, num_classes, device)

        self._base = IntegratedGradients(
            nn.Sequential(
                model,
                LogitGapWrtTarget(num_classes=num_classes),
            )
        )
        self.num_steps = 50

    def attribute(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        x = x.to(self.device)
        y = y.to(self.device)

        attribution_map = self._base.attribute(x, target=y, n_steps=self.num_steps)

        with torch.no_grad():
            logit = self.model(x).detach().cpu().numpy()

        return logit, attribution_map.detach().cpu().numpy().sum(axis=1)


def generate_superpixel_mask(
    input_size: Tuple[int, int], patch_size: int
) -> npt.NDArray:
    # we assume that
    # - 1) input_size.h = input_size.w
    # - 2) iput_size.h % patch_size == 0
    assert input_size[0] == input_size[1]
    assert input_size[0] % patch_size == 0

    size = input_size[0]

    mask = np.zeros(input_size)

    total_patch_steps = size // patch_size

    # vertical
    for step_j in range(total_patch_steps):
        # horizontal
        for step_i in range(total_patch_steps):
            patch_id = step_j * total_patch_steps + step_i

            assert 0 <= patch_id <= total_patch_steps**2 - 1

            mask[
                step_j * patch_size : (step_j + 1) * patch_size,
                step_i * patch_size : (step_i + 1) * patch_size,
            ] = patch_id

    return mask


@register_explainer("svs")
class ShapleyValueSamplingExplainer(Explainer):
    def __init__(self, model: nn.Module, num_classes: int, device: str) -> None:
        super().__init__(model, num_classes, device)

        self._base = ShapleyValueSampling(
            nn.Sequential(
                model,
                LogitGapWrtTarget(num_classes=num_classes),
            )
        )
        self.n_samples = 25
        self.patch_size = 8

    def attribute(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            logit = self.model(x).detach().cpu().numpy()

        feature_mask = generate_superpixel_mask(
            input_size=(x.shape[2], x.shape[3]), patch_size=self.patch_size
        )
        feature_mask = torch.from_numpy(feature_mask).to(self.device)  # type: ignore

        attribution_map = self._base.attribute(
            x, target=y, feature_mask=feature_mask, n_samples=self.n_samples
        )

        return logit, attribution_map.detach().cpu().numpy().sum(axis=1)


def get_explainer(
    name: str,
    model: nn.Module,
    num_classes: int,
    device: str,
) -> Explainer:
    return EXPLAINERS[name](model=model, num_classes=num_classes, device=device)
