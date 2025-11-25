from abc import abstractmethod
import typing
import torch
from torch.utils.data import Subset
from torchvision import datasets as tvd


class WithValidationSetMixin:
    @abstractmethod
    def create_train_val_split(
        self,
        rng: torch.Generator,
    ) -> typing.Tuple[Subset[tvd.VisionDataset], Subset[tvd.VisionDataset]]:
        pass
