from abc import ABC

import typing
import numpy.typing as npt

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms

from zennit.torchvision import ResNetCanonizer
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient

from tqdm import tqdm

from xaikd import utils
from xaikd import datasets


def make_attributor_for(
    model: nn.Module,
    input_statistics: typing.Tuple[typing.Tuple[float, ...], typing.Tuple[float, ...]],
) -> Gradient:
    # remark this only works for cifar10 and cifar100 for now
    assert type(model) == models.resnet.ResNet

    input_transform = transforms.Normalize(*input_statistics)

    low, high = input_transform(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))

    canonizer = ResNetCanonizer()

    composite = EpsilonGammaBox(low=low, high=high, canonizers=[canonizer])

    return Gradient(model=model, composite=composite)


class LogitModifier(ABC):
    def __call__(
        self, logits: torch.Tensor, targets: typing.Union[torch.Tensor, None]
    ) -> torch.Tensor:
        raise NotImplemented


class OneClassEvidence(LogitModifier):
    def __init__(self, dataset: datasets.DatasetConfiguration) -> None:
        self.dataset = dataset

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.clone()
        return logits * F.one_hot(targets, self.dataset.num_classes).to(logits.device)


class LogOddEvidence(LogitModifier):
    def __init__(
        self, classes: typing.List[int], dataset: datasets.TwoClassesDataset
    ) -> None:
        # todo: perhaps, we only need `dataset` and extract classes from there.
        assert len(classes) == 2

        assert isinstance(dataset, datasets.TwoClassesDataset)

        self.classes = classes
        self.dataset = dataset

    def __call__(self, logits: torch.Tensor, targets=None) -> torch.Tensor:
        output = torch.zeros_like(logits)
        # todo: this that assinging with targets has no effect
        output[:, self.classes[0]] = logits[:, self.classes[0]]
        output[:, self.classes[1]] = -logits[:, self.classes[1]]

        return output


def extract_activation_context(
    model: nn.Module,
    layer: str,
    dataset: datasets.DatasetConfiguration,
    logit_modifier: LogitModifier,
    device="cpu",
    number_of_selected_spatial_locations=20,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    data_loader = dataset.loader(train_split=True)

    arr_act = []
    arr_ctx = []

    try:
        module, hook = utils.interceptor.attach_hook_intercept_output(model, layer)

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
                    act, ctx, num_locations=number_of_selected_spatial_locations
                )
                arr_act.append(selected_act)
                arr_ctx.append(selected_ctx)

    finally:
        hook.remove()

    print(f"{layer}: output-dims={output_dimensions}")

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    return arr_act, arr_ctx
