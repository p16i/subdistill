import os
from pathlib import Path

PACKAGE_DIR = Path(os.path.dirname(__file__))

BASIS_NAMES = [
    "pca--centered",
    "prca--centered",
    "prca-abs--centered",
]

LAMBDA_LAYER_FOR_POLICIES = {
    "basis-identity:pca--uncentered": 1,
    "basis-identity:prca-sortabs--uncentered": 1,
    "basis-identity:random--uncentered": 1e4,
    "vid": 1e6,
    "attention-transfer": 0.1,
    "fitnet": 1,
    "fitnet-2l": 1,
    "fitnet-3l": 1,
}

ARCH_LAYER_DIMENSIONS = dict(
    dict(
        resnet18={
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
            "layer4.0": 512,
            "layer4.1": 512,
        },
        resnet50={
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
            "layer4.0": 2048,
            "layer4.1": 2048,
            "layer4.2": 2048,
        },
        vgg11={
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
            "layer5": 512,
        },
        vgg16={
            "layer1": 64,
            "layer2": 128,
            "layer3": 256,
            "layer4": 512,
            "layer5": 512,
        },
    )
)
