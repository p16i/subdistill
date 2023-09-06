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
        resnet18compr2={
            "layer1": 64 // 2,
            "layer2": 128 // 2,
            "layer3": 256 // 2,
            "layer4": 512 // 2,
            "layer4.0": 512 // 2,
            "layer4.1": 512 // 2,
        },
        resnet18compr4={
            "layer1": 64 // 4,
            "layer2": 128 // 4,
            "layer3": 256 // 4,
            "layer4": 512 // 4,
        },
        resnet18compr8={
            "layer1": 64 // 8,
            "layer2": 128 // 8,
            "layer3": 256 // 8,
            "layer4": 512 // 8,
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
