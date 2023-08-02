from typing import Callable
from torchvision import transforms

from xaikd import datasets


def get_augmentation_for(dataset: datasets.DatasetConfiguration) -> Callable:
    if isinstance(dataset, datasets.CIFAR10) or isinstance(
        dataset, datasets.Cifar100SuperClassesDataset
    ):
        return [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    # elif "imagenet" in dataset_name:
    #     return [
    #         transforms.RandomHorizontalFlip(),
    #     ]
    else:
        raise ValueError(f"We don't have any data augmentation for `{dataset}`.")
