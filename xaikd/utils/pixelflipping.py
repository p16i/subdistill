import torch
from typing import Tuple

from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
from numpy import typing as npt

from PIL import Image
import cv2


def generate_perturbation_masks(
    heatmap: npt.NDArray, pooled_heatmap_size=(14, 14), image_size=(224, 224)
) -> npt.NDArray:

    assert len(heatmap.shape) == 2 and heatmap.shape[0] == heatmap.shape[1]

    # convert heatmap to size
    pooled_heatmap = (
        F.adaptive_avg_pool2d(
            torch.from_numpy(heatmap).unsqueeze(0), output_size=pooled_heatmap_size
        )
        .squeeze(0)
        .numpy()
    )

    # Take most relevant first.
    indices = np.argsort(-pooled_heatmap.reshape(-1))
    coors = np.array(np.unravel_index(indices, pooled_heatmap.shape)).T.tolist()

    mask = np.zeros(image_size, dtype=np.uint8)

    patch_size = image_size[0] // pooled_heatmap_size[0]

    arr_masks = [mask.copy()]
    while len(coors) > 0:
        _coor = coors.pop(0)

        mask[
            _coor[0] * patch_size : (_coor[0] + 1) * patch_size,
            _coor[1] * patch_size : (_coor[1] + 1) * patch_size,
        ] = 1

        arr_masks.append(mask.copy())

    arr_masks = np.stack(arr_masks)

    np.testing.assert_equal(
        arr_masks.shape[0],
        np.prod(pooled_heatmap_size) + 1,
    )

    return arr_masks


def perturb_and_inpaint(
    img: Image.Image, perturbing_mask: npt.NDArray, baseline: npt.NDArray, radius=16
) -> Image.Image:

    assert isinstance(img, Image.Image)

    # add channel dimension
    perturbing_mask = perturbing_mask[:, :, np.newaxis]

    perturbed_img = img * (1 - perturbing_mask) + baseline * perturbing_mask

    # Importan Remark: cv2 uses "BGR" color format while PILLOW uses "RGB"
    # Ref: https://stackoverflow.com/a/48602446
    perturbed_img_cv_bgr = cv2.cvtColor(np.asarray(perturbed_img), cv2.COLOR_RGB2BGR)

    inpainted_perturbed_img_cv_bgr = cv2.inpaint(
        perturbed_img_cv_bgr, perturbing_mask.squeeze(), radius, cv2.INPAINT_TELEA
    )

    inpainted_perturbed_img_cv_rgb = cv2.cvtColor(
        inpainted_perturbed_img_cv_bgr, cv2.COLOR_BGR2RGB
    )

    inpainted_perturbed_img_pil_rgb = Image.fromarray(inpainted_perturbed_img_cv_rgb)

    assert isinstance(inpainted_perturbed_img_pil_rgb, Image.Image)

    return inpainted_perturbed_img_pil_rgb


@torch.no_grad()
def perform_pixel_flipping(
    model: nn.Module,
    img: Image.Image,
    heatmap: npt.NDArray,
    target: int,
    baseline: npt.NDArray,
    transform: transforms.Compose,
    device="cpu",
) -> Tuple[npt.NDArray, npt.NDArray]:

    arr_masks = generate_perturbation_masks(heatmap)
    n_steps = arr_masks.shape[0]
    arr_num_flipped_pixels = np.sum(arr_masks.reshape((n_steps, -1)), axis=1)

    assert arr_num_flipped_pixels[0] == 0
    assert arr_num_flipped_pixels[-1] == np.prod(img.size)

    arr_img_tensors = []
    for step_ix in range(n_steps):
        mask = arr_masks[step_ix]

        pertubed_img = perturb_and_inpaint(img, mask, baseline=baseline)
        arr_img_tensors.append(transform(pertubed_img))

    x = torch.stack(arr_img_tensors).to(device)

    arr_logits = model(x).cpu().numpy()[:, target]

    return arr_num_flipped_pixels, arr_logits


def get_baseline_and_transform_for_perturbed_img(
    normalizer: transforms.Normalize,
) -> Tuple[npt.NDArray, transforms.Compose]:

    assert isinstance(normalizer, transforms.Normalize)

    # Remark: we have * 255 because we want to use it with PIL.Image whose
    # pixel values are in [0, 255].
    mean = np.array(normalizer.mean)
    baseline_value = (mean * 255).astype(np.uint8)

    return baseline_value, transforms.Compose([transforms.ToTensor(), normalizer])
