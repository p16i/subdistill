import typing

from functools import partial
from torch.utils.data import Dataset


from ..register import add_dataset_to_registry
from .original import ImageNet, IMAGENET_SUPERCLASS_MAPPING


class ImageNetSuperclassVsOthers(ImageNet):
    def __init__(self, super_class: str):
        super().__init__()

        self.selected_classes = IMAGENET_SUPERCLASS_MAPPING[super_class]
        self.num_classes = 1

    def transform_target(self, target: int) -> int:
        if target in self.selected_classes:
            return 1
        else:
            return 0

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:

        return super().create_subset(
            train_split=train_split, target_transform=self.transform_target
        )


def register_datasets():
    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        add_dataset_to_registry(
            f"imagenet-{superclass}-vs-others",
            partial(ImageNetSuperclassVsOthers, super_class=superclass),
        )


register_datasets()
