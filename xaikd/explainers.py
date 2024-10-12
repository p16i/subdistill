from numpy.typing import NDArray
from typing import Tuple
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T


from torchvision.models.mobilenetv3 import MobileNetV3

from zennit.attribution import Gradient

from xaikd import attributors, utils

from functools import partial

import numpy as np
from numpy import typing as npt


EXPLAINERS = dict()


def register_explainer(name):
    def wrapped(fn):
        EXPLAINERS[name] = fn

        return fn

    return wrapped


@register_explainer("lrp")
class Explainer:
    def __init__(self, model: nn.Module, input_transform: T.Normalize, **kwargs):

        low, high = input_transform(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))

        self.model = model
        self.composite = attributors.get_arch_specific_composite(
            model, lb=low, hb=high, **kwargs
        )

    def explain(
        self,
        dataloader: DataLoader,
        logit_modifier: attributors.LogitModifier,
        device="cpu",
    ) -> Tuple[npt.NDArray, npt.NDArray]:

        arr_heatmaps = []
        arr_logits = []

        with Gradient(self.model, self.composite) as attributor:
            for x, y in tqdm(dataloader):
                x = x.to(device)
                y = y.to(device)
                logits, heatmap = attributor.forward(
                    x, lambda logits: logit_modifier(logits, y)
                )

                logits = logits.detach().cpu().numpy()
                heatmap = heatmap.detach().cpu().numpy()

                heatmap = heatmap.sum(axis=1)

                arr_heatmaps.append(heatmap)
                arr_logits.append(logits)

        arr_heatmaps = np.vstack(arr_heatmaps)
        arr_logits = np.vstack(arr_logits)

        return arr_logits, arr_heatmaps


@register_explainer("random1")
class RandomExplainer(Explainer):
    def __init__(
        self, model: nn.Module, input_transform: T.Normalize, seed=1, **kwargs
    ):

        super().__init__(model, input_transform, **kwargs)
        self.seed = seed

    def explain(
        self,
        dataloader: DataLoader,
        logit_modifier: attributors.LogitModifier,
        device="cpu",
    ) -> Tuple[npt.NDArray, npt.NDArray]:

        rng = np.random.default_rng(seed=self.seed)

        arr_heatmaps = []
        arr_logits = []

        for x, _ in tqdm(dataloader):
            x = x.to(device)
            logits = self.model(x).detach().cpu().numpy()

            shape = x.shape

            heatmap = rng.random(shape)
            heatmap = heatmap.sum(axis=1)

            arr_logits.append(logits)
            arr_heatmaps.append(heatmap)

        arr_heatmaps = np.vstack(arr_heatmaps)
        arr_logits = np.vstack(arr_logits)

        return arr_logits, arr_heatmaps


@register_explainer("mobilenetlrp")
class MobileNetExplainer(Explainer):
    def __init__(self, model: nn.Module, input_transform: T.Normalize, **kwargs):

        assert isinstance(model, MobileNetV3)

        super().__init__(model, input_transform, **kwargs)

    def explain(
        self,
        dataloader: DataLoader,
        logit_modifier: attributors.LogitModifier,
        device="cpu",
    ) -> Tuple[npt.NDArray, npt.NDArray]:

        arr_logits = []
        arr_heatmaps = []

        with Gradient(self.model, self.composite) as attributor:
            for x, y in tqdm(dataloader):
                hook = None
                try:
                    x = x.to(device)
                    y = y.to(device)
                    module, hook = utils.interceptor.attach_hook_intercept_layer_output(
                        self.model,
                        "features.1",
                        should_retain_grad=True,
                        detach_output=False,
                    )
                    logits, _ = attributor.forward(
                        x, lambda logits: logit_modifier(logits, y)
                    )

                    logits = logits.detach().cpu().numpy()

                    # remark: we use heatmaps from this layer because the input heatmap is very noisy.
                    heatmap = (
                        utils.interceptor.get_output(module).grad.detach().cpu().numpy()
                    )

                    heatmap = heatmap.sum(axis=1)

                    arr_logits.append(logits)
                    arr_heatmaps.append(heatmap)

                finally:
                    if hook is not None:
                        hook.remove()

        arr_logits = np.vstack(arr_logits)
        arr_heatmaps = np.vstack(arr_heatmaps)

        return arr_logits, arr_heatmaps


def get_explainer(
    name: str, model: nn.Module, input_transform: T.Normalize, **kwargs
) -> Explainer:
    return EXPLAINERS[name](model=model, input_transform=input_transform, **kwargs)


def ano():
    for gamma in [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]:
        EXPLAINERS[f"lrp{gamma}"] = partial(Explainer, gamma=gamma)

        EXPLAINERS[f"mobilenetlrp{gamma}"] = partial(MobileNetExplainer, gamma=gamma)


ano()
