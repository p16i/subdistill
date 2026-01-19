import os
import io

from pathlib import Path

import numpy as np

from PIL import Image

from PIL.Image import Image as TypeImage

from torchvision.transforms import functional as F

from xaikd import constants, datasets
from torchvision.datasets import MNIST


DS_MNIST = MNIST(f"{datasets.DATADIR}", download=False)


# def imagenet_copyright(img: TypeImage, seed: int) -> TypeImage:
#     watermark = Image.open(
#         str(
#             Path(os.path.dirname(constants.PACKAGE_DIR))
#             / "resources"
#             / "copyright"
#             / "4.png"
#         )
#     )

#     rng = np.random.default_rng(seed=seed)

#     # this makes sure that we do NOT override the input image.
#     img = img.copy()

#     img_w, img_h = img.size
#     cw, ch = img_w // 2, img_h // 2
#     scale_size = 256
#     crop_size = 224

#     mw, mh = watermark.size

#     ratio = scale_size / mw

#     minsize = np.min([img_w, img_h])
#     ratio2 = minsize / scale_size

#     nmw = int(mw * ratio * ratio2)
#     nmh = int(mh * ratio * ratio2)
#     watermark = watermark.convert("L")
#     watermark = watermark.resize((nmw, nmh))

#     img.paste(
#         watermark,
#         (
#             cw + ax,
#             ch + ay,
#         ),
#     )

#     return img


def imagenet_copyright(img: TypeImage, seed: int) -> TypeImage:
    rng = np.random.default_rng(seed=seed)
    # location = tuple(rng.choice(constants.ARR_IMAGENET_COPYRIGHT2_CORNER_LOCATIONs))

    watermark = Image.open(
        str(
            Path(os.path.dirname(constants.PACKAGE_DIR))
            / "resources"
            / "copyright"
            / "4.png"
        )
    )

    # this makes sure that we do NOT override the input images
    img = img.copy()
    img_w, img_h = img.size

    cw, ch = img_w // 2, img_h // 2

    scale_size = 256
    crop_size = 224

    marksize = 100

    minsize = np.min([img_w, img_h])
    ratio2 = minsize / scale_size

    mw, mh = watermark.size

    nmw = int(marksize * ratio2)
    nmh = int((marksize / mw) * mh * ratio2)
    watermark = watermark.resize((nmw, nmh))

    # delta_x, delta_y = location

    adjust_with_crop = int(ratio2 * crop_size // 2)
    delta_x = rng.choice([-adjust_with_crop + 20, adjust_with_crop - nmw - 20])

    img.paste(
        watermark,
        (
            cw + delta_x,
            ch + adjust_with_crop - nmh - 10,
        ),
        mask=watermark.point(lambda i: 0.9 * i),
    )

    return img


def imagenet_center_watermark(img: TypeImage) -> TypeImage:
    watermark = Image.open(
        str(
            Path(os.path.dirname(constants.PACKAGE_DIR))
            / "resources"
            / "copyright"
            / "3.png"
        )
    )

    # this makes sure that we do NOT override the input images
    img = img.copy()
    img_w, img_h = img.size
    cw = int(img_w // 2)
    ch = int(img_h * (4 / 5))
    img_w, img_h = img.size

    scale_size = 256
    marksize = 120

    mw, mh = watermark.size

    minsize = np.min([img_w, img_h])
    ratio2 = minsize / scale_size

    nmw = int(marksize * ratio2)
    nmh = int((marksize / mw) * mh * ratio2)
    watermark = watermark.resize((nmw, nmh))

    img.paste(
        watermark,
        (
            cw - nmw // 2,
            ch - nmh // 2,
        ),
        mask=watermark.point(lambda i: 0.8 * i),
    )

    return img


def jpeg_artifact(img: TypeImage) -> TypeImage:
    # this makes sure that we do NOT override the input images
    img = img.copy()

    buffered = io.BytesIO()
    img.save(buffered, format="jpeg", optimize=True, quality=5)

    return Image.open(buffered)


def scaling_artifact(img: TypeImage, scaling_factor=4) -> TypeImage:
    h, w = img.size

    nh = h // scaling_factor
    nw = w // scaling_factor

    recon_img = F.resize(F.resize(img, size=[nh, nw]), size=[h, w])

    return recon_img


def mnist_corner(img: TypeImage, label: int, seed: int) -> TypeImage:
    rng = np.random.default_rng(seed=seed)
    img = img.copy()
    img_w, img_h = img.size
    cw, ch = img_w // 2, img_h // 2
    img_w, img_h = img.size

    marksize = 28 * 2

    arr_ix_with_same_label = np.argwhere(DS_MNIST.targets == label)

    ix = rng.choice(arr_ix_with_same_label.reshape(-1))
    img_mnist, _ = DS_MNIST[ix]

    watermark = img_mnist.convert("RGBA")

    watermark = watermark.resize((marksize, marksize))

    img.paste(
        watermark,
        (
            cw - (224 // 2),
            ch - (224 // 2),
        ),
        mask=watermark,
    )

    return img
