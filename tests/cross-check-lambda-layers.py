import pytest

import numpy as np

from xaikd import constants


pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    "teacher,config,expected",
    [
        ("imagenet-resnet18-tv", "imagenet-clean", (4, 0, 1, 1, 1, 2)),
        ("imagenet-resnet18-tv", "imagenet-small", (0, 2, 2, 1, 2, 2)),
        ("imagenet-resnet50-tv", "imagenet-clean", (3, 1, 0, 1, 2, 1)),
        (
            "imagenet-resnet50-tv",
            "imagenet-small",
            (1, 1, 3, 2, 1, 3),
        ),
        ("imagenet-vgg16-tv", "imagenet-clean", (0, 1, 1, 0, 1, 1)),
        ("imagenet-vgg16-tv", "imagenet-small", (5, 2, 2, 5, 1, 2)),
        ("imagenet-nfnetf0-dm", "imagenet-clean", (1, 2, 1, 3, 5, 2)),
        ("imagenet-nfnetf0-dm", "imagenet-small", (2, 2, 4, 4, 5, 2)),
        ("imagenet-vitb-tv", "imagenet-clean", (0, 3, 3, 2, None, 1)),
        ("imagenet-vitb-tv", "imagenet-small", (0, 4, 3, 2, None, 2)),
    ],
)
@pytest.mark.parametrize(
    "pix,policy",
    list(
        enumerate(
            [
                "fitnet",
                "attention-transfer",
                "vid",
                "basis-identity:pca--uncentered",
                "basis-identity:prca-sortabs--uncentered",
                "basis-identity:pcalookahead--uncentered",
            ]
        )
    ),
)
def test(teacher, config, expected, pix, policy):
    if teacher == "imagenet-vitb-tv" and "prca" in policy:
        pytest.skip("unsupported configuration")

    if policy == "fitnet":
        if teacher in ["imagenet-nfnetf0-dm", "imagenet-vitb-tv"]:
            policy = "fitnet-noact"
        else:
            policy = "fitnet-relu"

    np.testing.assert_equal(
        constants.DEFAULT_LAMBDA_LAYER[config][teacher][policy],
        np.power(10, expected[pix]),
    )
