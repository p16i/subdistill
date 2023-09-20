import os
from pathlib import Path

PACKAGE_DIR = Path(os.path.dirname(__file__))

BASIS_NAMES = [
    "pca--centered",
    "prca--centered",
    "prca-abs--centered",
]

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
