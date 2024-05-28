import os
from pathlib import Path
import pandas as pd

PACKAGE_DIR = Path(os.path.dirname(__file__))

ARR_STUDENT_DIMENSIONS = [
    (32, 24, 16, 8),
    (40, 32, 24, 16),
    (48, 40, 32, 24),
    (56, 48, 40, 32),
]

STUDENT_MODEL_FOR_TESTING = "student-32-24-16-8"

TRAINING_VAL_SPLIT_RATIO = 0.8


CIFAR100_SUPER_CLASS_MAPPING = PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv"

CIFAR100_SUPER_CLASSES = (
    pd.read_csv(CIFAR100_SUPER_CLASS_MAPPING)[
        "coarse_label_name"
    ]
    .unique()
    .tolist()
)

# BASIS_NAMES = [
#     "pca--centered",
#     "prca--centered",
#     "prca-abs--centered",
# ]

# LAMBDA_LAYER_FOR_POLICIES = {
#     "nothing": {"default": 0},
#     "basis-identity:pca--uncentered": {
#         "resnet18xscifarcompr1": 1000.0,
#         "resnet18xscifarcompr2": 1000.0,
#         "resnet18xscifarcompr4": 1000.0,
#         "resnet18dims32-24-24-5": 1000.0,
#         "resnet18dims24-16-16-5": 1000.0,
#         "resnet18dims16-8-8-5": 1000.0,
#         "resnet18dims64-48-48-10": 1,
#         "resnet18dims48-32-32-10": 1,
#         "resnet18dims32-16-16-10": 1,
#     },
#     "basis-identity:prca-sortabs--uncentered": {
#         "resnet18xscifarcompr1": 1000.0,
#         "resnet18xscifarcompr2": 1000.0,
#         "resnet18xscifarcompr4": 1000.0,
#         "resnet18dims32-24-24-5": 1000.0,
#         "resnet18dims24-16-16-5": 1000.0,
#         "resnet18dims16-8-8-5": 1000.0,
#         "resnet18dims64-48-48-10": 1,
#         "resnet18dims48-32-32-10": 1,
#         "resnet18dims32-16-16-10": 1,
#     },
#     "basis-identity:random--uncentered": {
#         "resnet18xscifarcompr1": 1000.0,
#         "resnet18xscifarcompr2": 1000.0,
#         "resnet18xscifarcompr4": 1,
#         "resnet18dims32-24-24-5": 1000.0,
#         "resnet18dims24-16-16-5": 1000.0,
#         "resnet18dims16-8-8-5": 1000.0,
#         "resnet18dims64-48-48-10": 1,
#         "resnet18dims48-32-32-10": 1,
#         "resnet18dims32-16-16-10": 1,
#     },
#     "vid": {
#         "resnet18xscifarcompr1": 1000000.0,
#         "resnet18xscifarcompr2": 1000000.0,
#         "resnet18xscifarcompr4": 100000.0,
#         "resnet18dims32-24-24-5": 1000000.0,
#         "resnet18dims24-16-16-5": 1e6,
#         "resnet18dims16-8-8-5": 1e6,
#         "resnet18dims64-48-48-10": 1,
#         "resnet18dims48-32-32-10": 1,
#         "resnet18dims32-16-16-10": 1,
#     },
#     "attention-transfer": {
#         "resnet18xscifarcompr1": 1000.0,
#         "resnet18xscifarcompr2": 10.0,
#         "resnet18xscifarcompr4": 10.0,
#     },
#     "fitnet": {
#         "resnet18xscifarcompr1": 0.01,
#         "resnet18xscifarcompr2": 0.001,
#         "resnet18xscifarcompr4": 0.001,
#     },
# }


# def get_lamba_layer_for_policy_student(policy: str, student: str) -> float:
#     policy_lambda_layers: dict = LAMBDA_LAYER_FOR_POLICIES[policy]

#     key = student if student in policy_lambda_layers else "default"

#     lambda_layer = policy_lambda_layers[key]

#     print(f"Retrieving `lambda_layer`(policy=`{policy}`, key=`{key}`)={lambda_layer}")

#     return lambda_layer


ARCH_LAYER_DIMENSIONS = dict(
    resnet18={
        "layer1": 64,
        "layer2": 128,
        "layer3": 256,
        "layer4": 512,
        # "layer4.0": 512,
        # "layer4.1": 512,
    },
    resnet34={"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512},
    resnet50={
        "layer1": 256,
        "layer2": 512,
        "layer3": 1024,
        "layer4": 2048,
        # "layer4.0": 2048,
        # "layer4.1": 2048,
        # "layer4.2": 2048,
    },
    resnet101={"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    resnet152={"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048},
    vgg11={
        "features.5": 128,
        "features.10": 256,
        "features.15": 512,
        "features.20": 512,
    },
    vgg11bn={
        "features.7": 128,
        "features.14": 256,
        "features.21": 512,
        "features.28": 512,
    },
    vgg13={
        "features.9": 128,
        "features.14": 256,
        "features.19": 512,
        "features.24": 512,
    },
    vgg13bn={
        "features.13": 128,
        "features.20": 256,
        "features.27": 512,
        "features.34": 512,
    },
    vgg16={
        "features.9": 128,
        "features.16": 256,
        "features.23": 512,
        "features.30": 512,
    },
    vgg16bn={
        "features.13": 128,
        "features.23": 256,
        "features.33": 512,
        "features.43": 512,
    },
    vgg19={
        "features.9": 128,
        "features.18": 256,
        "features.27": 512,
        "features.36": 512,
    },
    vgg19bn={
        "features.13": 128,
        "features.26": 256,
        "features.39": 512,
        "features.52": 512,
    },
    nfnetf0={
        "stages.0": 256,
        "stages.1": 512,
        "stages.2": 1536,
        "stages.3": 1536,
    },
)
