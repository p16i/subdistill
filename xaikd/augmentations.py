import typing
from torchvision import transforms

from xaikd import datasets


def get_augmentation_for(
    dataset: datasets.DatasetConfiguration,
) -> typing.List[typing.Callable]:
    raise NotImplementedError("obsolete")
    if isinstance(dataset, datasets.CIFAR10) or isinstance(
        dataset, datasets.Cifar100SuperClassesDataset
    ):
        # ref: to add
        return [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif isinstance(dataset, datasets.ImageNet) or isinstance(
        dataset, datasets.ImageNetButterfly
    ):
        # ref: https://github.com/pytorch/vision/blob/v0.10.0/references/classification/presets.py#L12
        return [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        ]
    else:
        raise ValueError(f"We don't have any data augmentation for `{dataset}`.")
