from abc import ABC

import typing
import numpy.typing as npt

import numpy as np

import torch

import timm
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader

from zennit.composites import Composite
from zennit.canonizers import Canonizer
from zennit.torchvision import ResNetCanonizer, VGGCanonizer
from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient

from tqdm.auto import tqdm

from xaikd import utils
from xaikd import datasets


from xaikd.models.interfaces import DistillableModel

import zennit

from xaikd import lrp

from xaikd.logit_modifiers import LogitModifier


from functools import partial


def get_arch_specific_composite(
    model: nn.Module, lb: torch.Tensor, hb: torch.Tensor, **kwargs
) -> Composite | None:
    if isinstance(model, models.resnet.ResNet):
        return EpsilonGammaBox(
            low=lb, high=hb, canonizers=[ResNetCanonizer()], **kwargs
        )
    elif isinstance(model, torchvision.models.vgg.VGG):
        if utils.modules.has_batchnorm(model):
            return EpsilonGammaBox(
                low=lb, high=hb, canonizers=[VGGCanonizer()], **kwargs
            )
        else:
            return EpsilonGammaBox(low=lb, high=hb, canonizers=[], **kwargs)
    elif isinstance(model, timm.models.nfnet.NormFreeNet):
        return lrp.nfnets._build_composite(lb=lb, hb=hb, **kwargs)
    elif isinstance(model, torchvision.models.VisionTransformer):
        return lrp.vit._build_composite(lb=lb, hb=hb, **kwargs)
    elif isinstance(model, torchvision.models.MobileNetV3):
        return lrp.mobilenets._build_composite(lb=lb, hb=hb, **kwargs)
    else:
        raise NotImplementedError("")


def make_attributor_for(
    model: nn.Module,
    input_statistics: typing.Tuple[typing.Tuple[float, ...], typing.Tuple[float, ...]],
) -> Gradient:
    input_transform = transforms.Normalize(*input_statistics)

    low, high = input_transform(torch.tensor([[[[[0.0]]] * 3], [[[[1.0]]] * 3]]))

    composite = get_arch_specific_composite(model, lb=low, hb=high)

    return Gradient(model=model, composite=composite)


def extract_activation_context(
    model: nn.Module,
    layer: str,
    dataset: datasets.DatasetConfiguration,
    data_loader: DataLoader,
    logit_modifier: LogitModifier,
    rng: np.random.Generator,
    device="cpu",
    number_of_selected_spatial_locations=20,
    strict_mode=False,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    arr_act = []
    arr_ctx = []

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=True, detach_output=False
        )

        with make_attributor_for(model, dataset.input_statistics) as attributor:
            for batch in tqdm(data_loader):
                x, y = batch
                x = x.to(device)

                _ = attributor.forward(x, lambda logits: logit_modifier(logits, y))

                act = utils.interceptor.get_output(module)

                assert act.grad is not None
                rel = act.grad

                output_dimensions = act.shape[1:]

                ctx = torch.where(act.abs() > 0, rel / act, 0)

                if strict_mode:
                    np.testing.assert_allclose(
                        (act * ctx).detach().cpu().numpy(),
                        rel.detach().cpu().numpy(),
                        atol=1e-6,
                    )

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


def extract_activation_grad(
    model: nn.Module,
    layer: str,
    dataloader: DataLoader,
    logit_modifier: LogitModifier,
    rng: np.random.Generator,
    device="cpu",
    number_of_selected_spatial_locations=20,
    verbose=False,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    arr_act = []
    arr_ctx = []

    try:
        module, hook = utils.interceptor.attach_hook_intercept_layer_output(
            model, layer, should_retain_grad=True, detach_output=False
        )
        # todo: is the order between hook and forward matter?
        for batch in tqdm(dataloader, desc=f"extract act-grad at layer={layer}"):
            x, y = batch
            x = x.to(device)

            logits = model(x)

            logit_modifier(logits=logits, targets=y).sum().backward()

            act = utils.interceptor.get_output(module)

            assert act.grad is not None
            ctx = act.grad

            output_dimensions = act.shape[1:]

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

    if verbose:
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
