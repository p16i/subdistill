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
    pd.read_csv(CIFAR100_SUPER_CLASS_MAPPING)["coarse_label_name"].unique().tolist()
)


DEFAULT_TEACHER_STUDENT_LAYER_MAPPING = {
    "cifar100-resnet18-v1": "layer3:layer3,layer4:layer4",
    "imagenet-resnet18-tv": "layer3:layer3,layer4:layer4",
    "imagenet-resnet50-tv": "layer3:layer3,layer4:layer4",
    "imagenet-vgg16-tv": "features.23:layer3,features.30:layer4",
    "imagenet-nfnetf0-dm": "stages.2:layer3,stages.3:layer4",
}

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


DEFAULT_LAMBDA_LAYER = {
    "dummy": {
        "policy-1": 0.1,
        "policy-2": 0.7,
        "vid": 0.8,
    },
    "cifar100-clean": {
        "fitnet": 1e5,
        "attention-transfer": 1e2,
        "vid": 1e2,
        "basis-identity:pca--uncentered": 1e2,
        "basis-identity:prca-sortabs--uncentered": 1e1,
        "basis-identity:pcalookahead--uncentered": 1e1,
    },
    "cifar100-small": {
        "fitnet": 1e1,
        "attention-transfer": 1e2,
        "vid": 1e3,
        "basis-identity:pca--uncentered": 1e1,
        "basis-identity:prca-sortabs--uncentered": 1e1,
        "basis-identity:pcalookahead--uncentered": 1e2,
    },
    "cifar100-spurious": {
        "fitnet": 1e5,
        "attention-transfer": 1e4,
        "vid": 1e5,
        "basis-identity:pca--uncentered": 1e4,
        "basis-identity:prca-sortabs--uncentered": 1e4,
        "basis-identity:pcalookahead--uncentered": 1e4,
    },
}
