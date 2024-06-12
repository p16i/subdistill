import os
from pathlib import Path

import numpy as np
from PIL import Image

from xaikd import constants


def imagenet_copyright(img: Image, watermark: Image) -> Image:
    # this makes sure that we do NOT override the input image.
    img = img.copy()

    img_w, img_h = img.size
    cw, ch = img_w // 2, img_h // 2
    scale_size = 256
    crop_size = 224

    mw, mh = watermark.size

    ratio = scale_size / mw

    minsize = np.min([img_w, img_h])
    ratio2 = minsize / scale_size

    nmw = int(mw * ratio * ratio2)
    nmh = int(mh * ratio * ratio2)
    watermark = watermark.convert("L")
    watermark = watermark.resize((nmw, nmh))

    img.paste(
        watermark,
        (
            cw - nmw // 2,
            ch + int(ratio2 * crop_size // 2) - nmh,
        ),
    )

    return img


def apply_copyright2_to_image(img: Image, rng: np.random.Generator) -> Image:
    location = tuple(rng.choice(constants.ARR_IMAGENET_COPYRIGHT2_CORNER_LOCATIONs))

    watermark = Image.open(
        str(
            Path(os.path.dirname(constants.PACKAGE_DIR))
            / "resources"
            / "copyright"
            / "2.png"
        )
    )

    # this makes sure that we do NOT override the input images
    img = img.copy()
    img_w, img_h = img.size

    cw, ch = img_w // 2, img_h // 2

    scale_size = 256

    marksize = 150

    minsize = np.min([img_w, img_h])
    ratio2 = minsize / scale_size

    mw, mh = watermark.size

    nmw = int(marksize * ratio2)
    nmh = int((marksize / mw) * mh * ratio2)
    watermark = watermark.resize((nmw, nmh))

    delta_x, delta_y = location
    img.paste(
        watermark,
        (
            cw - nmw // 2 + delta_x,
            ch - nmh // 2 + delta_y,
        ),
        mask=watermark,
    )

    return img


def imagenet_watermark(img: Image) -> Image:
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
    cw, ch = img_w // 2, img_h // 2
    img_w, img_h = img.size

    scale_size = 256
    marksize = 150

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
        mask=watermark.point(lambda i: 0.5 * i),
    )

    return img
