import os
import typing

import numpy as np
import numpy.typing as npt

import torch
from torch import nn
from tqdm import tqdm

from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient

from . import data

from xaikd import attributors, utils

COMPOSITE = EpsilonGammaBox(low=torch.tensor([-1] * 3), high=torch.tensor([1] * 3))


def extract_activation_context(
    model: nn.Module,
    module: nn.Module,
    dataset: data.Dataset,
    selected_classes: typing.Tuple[int, int],
    total_classes=data.NUM_CLASSES,
    output_dir="./tmp",
    seed=1,
) -> typing.Tuple[npt.NDArray, npt.NDArray]:
    assert len(selected_classes) == 2

    other_classes = list(set(range(total_classes)).difference(selected_classes))

    print(f"We focus on these classes: {selected_classes} from {total_classes} classes")
    print(f"and discarded classes: {other_classes}")

    slug = "class--" + "vs".join(np.array(selected_classes).astype(str))

    arr_act = []
    arr_ctx = []

    np.random.seed(seed)

    print(f"Extracting activation from {module}")

    output_modifier = attributors.LogOddEvidence(selected_classes)

    train_dl, _ = data.build_subset_loaders(dataset, selected_classes)

    try:
        module, hook = utils.interceptor.attach_hook_intercept_module(module)

        with Gradient(model=model, composite=COMPOSITE) as attributor:
            for bix, batch in tqdm(enumerate(train_dl)):
                x, y = batch

                assert np.isin(
                    y.numpy(), selected_classes
                ).all(), f"{selected_classes} :: {y}"

                output, attribution = attributor(
                    x, lambda output: output_modifier(output, y)
                )

                act = getattr(module, "__output")

                output_dimensions = act.shape

                if bix == 0:
                    print(f">  dimensions: {output_dimensions}")

                delattr(module, "__output")

                rel = act.grad

                ctx = torch.where(act.abs() > 0, rel / act, 0)

                assert torch.allclose(act * ctx, rel)

                assert ctx.shape == act.shape
                selected_act = act.detach().cpu().numpy()
                selected_ctx = ctx.detach().cpu().numpy()

                arr_act.append(selected_act)
                arr_ctx.append(selected_ctx)

    finally:
        hook.remove()

    arr_act = np.vstack(arr_act)
    arr_ctx = np.vstack(arr_ctx)

    return arr_act, arr_ctx
    # print(f"> (shape: {arr_act.shape})")

    # n, d = arr_act.shape

    # output_dir = output_dir / f"layer-{layer}" / slug

    # print(f"Output: {output_dir}")

    # os.makedirs(output_dir, exist_ok=True)

    # np.save(f"{output_dir}/act", arr_act)
    # np.save(f"{output_dir}/ctx", arr_ctx)
    # np.save(f"{output_dir}/act_mean", np.mean(arr_act, axis=0))

    # return output_dir
