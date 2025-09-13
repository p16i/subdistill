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

STUDENT_MODEL_FOR_TESTING = "student-32-24-16-8"

DEFAULT_BATCH_SIZE = 64

TRAINING_VAL_SPLIT_RATIO = 0.9


CIFAR100_SUPER_CLASS_MAPPING = PACKAGE_DIR / "resources" / "cifar100-label-mapping.csv"

CIFAR100_SUPER_CLASSES = (
    pd.read_csv(CIFAR100_SUPER_CLASS_MAPPING)["coarse_label_name"].unique().tolist()
)

ARR_IMAGENET_COPYRIGHT2_CORNER_LOCATIONs = list(
    itertools.product([-40, 40], [-100, 100])
)
