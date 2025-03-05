import typing


from torchvision import transforms
from torchvision import datasets as tvd

from .. import (
    TORCHVISION_DATASET_DOWNLOAD,
    DATADIR,
    DatasetConfiguration,
)
from ..register import register_dataset


@register_dataset("cifar100")
class CIFAR100(DatasetConfiguration):
    selected_classes = list(range(100))

    def __init__(self):
        # ref: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py#L12C1-L13C83
        self._normalizer = transforms.Normalize(
            mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        )

        self.input_transformation = transforms.Compose(
            [transforms.ToTensor(), self._normalizer]
        )

        # ref: https://github.com/zju-vipa/NetGraft/blob/main/utils/data.py#L35
        self.input_training_transformation = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalizer,
            ]
        )

        self.num_classes = 100
        # remark: we use the transformation (Normalization) of CIFAR10 here!
        self.dataclass = tvd.CIFAR100
        self.root = DATADIR / "cifar100"

    def transform_target(self, target: int) -> int:
        return target

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> tvd.CIFAR100:
        return self.dataclass(
            root=self.root,
            train=train_split,
            transform=self.input_transformation,
            download=TORCHVISION_DATASET_DOWNLOAD,
            target_transform=target_transform,
        )
