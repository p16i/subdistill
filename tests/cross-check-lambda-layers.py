import pytest

import numpy as np

from xaikd import constants


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
    ],
)
@pytest.mark.parametrize(
    "pix,policy",
    list(
        enumerate(
            [
                "fitnet-relu",
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

    np.testing.assert_equal(
        constants.DEFAULT_LAMBDA_LAYER[config][teacher][policy],
        np.power(10, expected[pix]),
    )
