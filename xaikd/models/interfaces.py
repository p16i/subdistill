import typing
from abc import ABC, abstractmethod

from torch import nn


class SplitableMixin(ABC):
    @abstractmethod
    def split_at(self, layer: str) -> typing.Tuple[nn.Module, nn.Module, nn.Module]:
        pass


class DistillableModel(nn.Module, SplitableMixin):
    pass

    @classmethod
    @abstractmethod
    def cast(cls, obj) -> typing.Self:
        pass


# approximator generator?
