from abc import ABC, abstractmethod
from typing import Tuple
import numpy.typing as npt


import torch
from torch import nn

from torch.nn import functional as F


from zennit.attribution import Gradient

import numpy as np
from numpy import typing as npt

from captum.attr import IntegratedGradients, ShapleyValueSampling
from zennit.torchvision import ResNetCanonizer
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient

from torchvision import transforms as T

EXPLAINERS = dict()

NORMALIZER = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


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

        self.gamma = 10

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
            target = F.one_hot(y, num_classes=self.num_classes).float()

            target = target.to(self.device)
            x = x.to(self.device)

            logit, attribution = attributor(x, target)  # type: ignore

        assert logit.shape == (nb, self.num_classes)
        logit = logit.detach().cpu().numpy()

        attribution = attribution.sum(dim=1).detach().cpu().numpy()
        assert attribution.shape == (nb, 224, 224), attribution.shape

        return logit, attribution


@register_explainer("intgrad")
class IntegratedGradientsExplainer(Explainer):
    def __init__(self, model: nn.Module, num_classes: int, device: str) -> None:
        super().__init__(model, num_classes, device)

        self._base = IntegratedGradients(model)
        self.num_steps = 100

    def attribute(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            logit = self.model(x).detach().cpu().numpy()

        attribution_map = self._base.attribute(x, target=y, n_steps=self.num_steps)

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

        self._base = ShapleyValueSampling(model)
        self.n_samples = 25
        self.patch_size = 16

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
            input_size=(x.shape[2], x.shape[3]), patch_size=16
        )
        feature_mask = torch.from_numpy(feature_mask).to(self.device)

        attribution_map = self._base.attribute(
            x, target=y, feature_mask=feature_mask, n_samples=self.n_samples
        )

        return logit, attribution_map.detach().cpu().numpy().sum(axis=0)


def get_explainer(
    name: str,
    model: nn.Module,
    num_classes: int,
    device: str,
) -> Explainer:
    return EXPLAINERS[name](model=model, num_classes=num_classes, device=device)
