import typing


from torchvision import transforms
from torchvision import datasets as tvd

from .. import (
    TORCHVISION_DATASET_DOWNLOAD,
    DATADIR,
    DatasetConfiguration,
)
from ..register import register_dataset


class CIFAR100Base(DatasetConfiguration):

    @property
    def _normalizer(self):
        # ref: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/conf/global_settings.py#L12C1-L13C83
        return transforms.Normalize(
            mean=(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            std=(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        )

    @property
    def input_transformation(self):
        return transforms.Compose([transforms.ToTensor(), self._normalizer])

    @property
    def input_training_transformation(self):
        # ref: https://github.com/zju-vipa/NetGraft/blob/main/utils/data.py#L35
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalizer,
            ]
        )

    @property
    def dataclass(self):
        return tvd.CIFAR100

    def create_subset(
        self,
        train_split=False,
    ) -> tvd.CIFAR100:
        return self.dataclass(
            root=str(DATADIR / "cifar100"),
            train=train_split,
            transform=self.input_transformation,
            download=TORCHVISION_DATASET_DOWNLOAD,
            target_transform=self.target_transform,
        )


@register_dataset("cifar100")
class CIFAR100(CIFAR100Base):

    @property
    def selected_classes(self):
        return list(range(100))

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def target_transform(self) -> typing.Union[None, typing.Callable]:
        return None
