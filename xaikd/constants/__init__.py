import os
from pathlib import Path
import pandas as pd

import itertools

from .lambda_layers import resolve_lambda_layer

PACKAGE_DIR = Path(os.path.dirname(__file__)).parent

ARR_STUDENT_DIMENSIONS = [
    (32, 24, 16, 8),
    (40, 32, 24, 16),
    (48, 40, 32, 24),
    (56, 48, 40, 32),
]

ARR_VIT_STUDENT_HIDDEN_DIMENSIONS = [
    48,
    60,
    72,
    132,
]


TRAINING_VAL_SPLIT_RATIO = 0.8


CIFAR100_SUPER_CLASS_MAPPING = PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv"

CIFAR100_SUPER_CLASSES = (
    pd.read_csv(CIFAR100_SUPER_CLASS_MAPPING)["coarse_label_name"].unique().tolist()
)


DEFAULT_TEACHER_STUDENT_LAYER_MAPPING = {
    "cifar100-resnet18-v1": "layer3:features.8,layer4:features.12",
    "imagenet-mobilenetl-tv": "features.12:features.8,features.16:features.12",
    "imagenet-resnet18-tv": "layer3:features.8,layer4:features.12",
    "imagenet-resnet50-tv": "layer3:features.8,layer4:features.12",
    "imagenet-vgg16-tv": "features.23:features.8,features.30:features.12",
    "imagenet-nfnetf0-dm": "stages.2:features.8,stages.3:features.12",
    "imagenet-vitb-tv": "encoder.layers.8:encoder.layers.2,encoder.layers.11:encoder.layers.3",
}

ARR_IMAGENET_COPYRIGHT2_CORNER_LOCATIONs = list(
    itertools.product([-40, 40], [-100, 100])
)
