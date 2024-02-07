import os
from pathlib import Path

PACKAGE_DIR = Path(os.path.dirname(__file__))

BASIS_NAMES = [
    "pca--centered",
    "prca--centered",
    "prca-abs--centered",
]

LAMBDA_LAYER_FOR_POLICIES = {
    "nothing": {"default": 0},
    "basis-identity:pca--uncentered": {
        "resnet18xscifarcompr1": 1000.0,
        "resnet18xscifarcompr2": 1000.0,
        "resnet18xscifarcompr4": 1000.0,
        "resnet18dims32-24-24-5": 1000.0,
        "resnet18dims24-16-16-5": 1000.0,
        "resnet18dims16-8-8-5": 1000.0,
        "resnet18dims64-48-48-10": 1,
        "resnet18dims48-32-32-10": 1,
        "resnet18dims32-16-16-10": 1,
    },
    "basis-identity:prca-sortabs--uncentered": {
        "resnet18xscifarcompr1": 1000.0,
        "resnet18xscifarcompr2": 1000.0,
        "resnet18xscifarcompr4": 1000.0,
        "resnet18dims32-24-24-5": 1000.0,
        "resnet18dims24-16-16-5": 1000.0,
        "resnet18dims16-8-8-5": 1000.0,
        "resnet18dims64-48-48-10": 1,
        "resnet18dims48-32-32-10": 1,
        "resnet18dims32-16-16-10": 1,
    },
    "basis-identity:random--uncentered": {
        "resnet18xscifarcompr1": 1000.0,
        "resnet18xscifarcompr2": 1000.0,
        "resnet18xscifarcompr4": 1,
        "resnet18dims32-24-24-5": 1000.0,
        "resnet18dims24-16-16-5": 1000.0,
        "resnet18dims16-8-8-5": 1000.0,
        "resnet18dims64-48-48-10": 1,
        "resnet18dims48-32-32-10": 1,
        "resnet18dims32-16-16-10": 1,
    },
    "vid": {
        "resnet18xscifarcompr1": 1000000.0,
        "resnet18xscifarcompr2": 1000000.0,
        "resnet18xscifarcompr4": 100000.0,
        "resnet18dims32-24-24-5": 1000000.0,
        "resnet18dims24-16-16-5": 1e6,
        "resnet18dims16-8-8-5": 1e6,
        "resnet18dims64-48-48-10": 1,
        "resnet18dims48-32-32-10": 1,
        "resnet18dims32-16-16-10": 1,
    },
    "attention-transfer": {
        "resnet18xscifarcompr1": 1000.0,
        "resnet18xscifarcompr2": 10.0,
        "resnet18xscifarcompr4": 10.0,
    },
    "fitnet": {
        "resnet18xscifarcompr1": 0.01,
        "resnet18xscifarcompr2": 0.001,
        "resnet18xscifarcompr4": 0.001,
    },
}


def get_lamba_layer_for_policy_student(policy: str, student: str) -> float:
    policy_lambda_layers: dict = LAMBDA_LAYER_FOR_POLICIES[policy]

    key = student if student in policy_lambda_layers else "default"

    lambda_layer = policy_lambda_layers[key]

    print(f"Retrieving `lambda_layer`(policy=`{policy}`, key=`{key}`)={lambda_layer}")

    return lambda_layer


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
