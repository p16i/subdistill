import typing

from functools import partial
from torch.utils.data import Dataset


from ..register import add_dataset_to_registry
from . import IMAGENET_SUPERCLASS_MAPPING
from .original import ImageNetBase


class ImageNetSuperclassVsOthers(ImageNetBase):

    def __init__(self, super_class: str):
        super().__init__()

        self._superclass = super_class
        self._selected_classes = IMAGENET_SUPERCLASS_MAPPING[super_class]

    @property
    def target_transform(self):
        def transform(target):

            if target in self.selected_classes:
                return 1
            else:
                return 0

        return transform

    @property
    def selected_classes(self) -> typing.List[int]:
        return self._selected_classes

    @property
    def num_classes(self) -> int:
        return 1


def register_datasets():
    for superclass in IMAGENET_SUPERCLASS_MAPPING.keys():
        add_dataset_to_registry(
            f"imagenet-{superclass}-vs-others",
            partial(ImageNetSuperclassVsOthers, super_class=superclass),
        )


register_datasets()
