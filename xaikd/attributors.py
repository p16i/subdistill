from abc import ABC

import typing
import numpy.typing as npt

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader

from zennit.torchvision import ResNetCanonizer
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient

from tqdm import tqdm

from xaikd import utils
from xaikd import datasets

from xaikd.models.interfaces import DistillableModel


def make_attributor_for(
    model: nn.Module,
    input_statistics: typing.Tuple[typing.Tuple[float, ...], typing.Tuple[float, ...]],
) -> Gradient:
    input_transform = transforms.Normalize(*input_statistics)

    low, high = input_transform(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))

    if isinstance(model, models.resnet.ResNet):
        canonizers = [ResNetCanonizer()]
    else:
        canonizers = []

    composite = EpsilonGammaBox(low=low, high=high, canonizers=canonizers)

    return Gradient(model=model, composite=composite)


class LogitModifier(ABC):
    def __call__(
        self, logits: torch.Tensor, targets: typing.Union[torch.Tensor, None]
    ) -> torch.Tensor:
        raise NotImplemented

    def __str__(self) -> str:
        raise NotImplemented


class OneClassEvidence(LogitModifier):
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        return logits * F.one_hot(targets, self.num_classes).to(logits.device)

    def __str__(self) -> str:
        return "oneclass"


class WinningClassEvidence(LogitModifier):
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        # todo: adding test
        wining_targets = torch.argmax(logits, dim=1)
        return logits * F.one_hot(wining_targets, self.num_classes).to(logits.device)

    def __str__(self) -> str:
        return "winingclass"


class ContrastTop2WinningClassesEvidence(LogitModifier):
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        # find the label of two winning classes
        _, indices = torch.topk(logits, dim=1, k=2)
        # todo: adding test
        return logits * F.one_hot(indices[:, 0], self.num_classes) - logits * F.one_hot(
            indices[:, 1], self.num_classes
        )

    def __str__(self) -> str:
        return "contrasttop2"


class OneClassLogSumExpEvidence(LogitModifier):
    def __init__(self, dataset: datasets.Cifar100SuperClassesDataset) -> None:
        self.dataset = dataset

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        logexp = torch.logsumexp(
            logits[:, self.dataset.selected_classes], dim=1, keepdim=True
        )
        return (logits - logexp) * F.one_hot(targets, self.dataset.num_classes).to(
            logits.device
        )

    def __str__(self) -> str:
        return "oneclasslogsum"


class SelectedClassesEvidence(LogitModifier):
    def __init__(self, selected_classes=typing.List[int]) -> None:
        self.selected_classes = selected_classes

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(logits)
        output[:, self.selected_classes] = logits[:, self.selected_classes]

        return output

    def __str__(self) -> str:
        return "selectedclasses"


class LogOddEvidence(LogitModifier):
    def __init__(
        self,
        classes: typing.Tuple[int, int],
    ) -> None:
        assert len(classes) == 2

        self.classes = classes

    def __call__(self, logits: torch.Tensor, targets=None) -> torch.Tensor:
        output = torch.zeros_like(logits)
        # todo: this that assinging with targets has no effect
        output[:, self.classes[0]] = logits[:, self.classes[0]]
        output[:, self.classes[1]] = -logits[:, self.classes[1]]

        return output

    def __str__(self) -> str:
        return "logodd"


def extract_activation_context(
    model: nn.Module,
    layer: str,
    dataset: datasets.DatasetConfiguration,
    data_loader: DataLoader,
    logit_modifier: LogitModifier,
    rng: np.random.Generator,
    device="cpu",
    number_of_selected_spatial_locations=20,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    arr_act = []
    arr_ctx = []

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=True
        )

        with make_attributor_for(model, dataset.input_statistics) as attributor:
            for batch in tqdm(data_loader):
                x, y = batch
                x = x.to(device)

                _ = attributor.forward(x, lambda logits: logit_modifier(logits, y))

                act = utils.interceptor.get_output(module)
                rel = act.grad

                output_dimensions = act.shape[1:]

                # todo: check this with Gregoire again!
                ctx = torch.where(act.abs() > 0, rel / act, 0)

                assert torch.allclose(act * ctx, rel)

                assert ctx.shape == act.shape

                act = act.detach().cpu().numpy()
                ctx = ctx.detach().cpu().numpy()

                selected_act, selected_ctx = utils.subsample_tensors(
                    act,
                    ctx,
                    num_locations=number_of_selected_spatial_locations,
                    rng=rng,
                )
                arr_act.append(selected_act)
                arr_ctx.append(selected_ctx)

    finally:
        hook.remove()

    print(f"{layer}: output-dims={output_dimensions}")

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    return arr_act, arr_ctx


def extract_activation(
    model: nn.Module,
    layer: str,
    dataset: datasets.DatasetConfiguration,
    data_loader: DataLoader,
    rng: np.random.Generator,
    device="cpu",
    number_of_selected_spatial_locations=20,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    arr_act = []

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=False
        )

        # todo(important): we don't need this composite context right?
        with make_attributor_for(model, dataset.input_statistics) as attributor:
            for batch in tqdm(data_loader):
                x, y = batch
                x = x.to(device)

                _ = model(x)

                act = utils.interceptor.get_output(module)

                act = act.detach().cpu().numpy()

                selected_act, _ = utils.subsample_tensors(
                    act,
                    act,
                    num_locations=number_of_selected_spatial_locations,
                    rng=rng,
                )
                arr_act.append(selected_act)

    finally:
        hook.remove()

    arr_act = np.vstack(arr_act)

    return arr_act
