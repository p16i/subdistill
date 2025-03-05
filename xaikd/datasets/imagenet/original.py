import typing
import numpy as np

from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import datasets as tvd

from torchvision.models import ResNet18_Weights


from ..register import register_dataset
from .. import DATADIR, DatasetConfiguration

DEFAULT_TRANSFORMATION = ResNet18_Weights.IMAGENET1K_V1.transforms()

# fmt: off
IMAGENET_SUPERCLASS_MAPPING = {
    "random": [100, 200, 300],  # for testing purpose
    "butterfly": [321, 322, 323, 324, 325, 326],
    "boat": [472, 554, 576, 625, 814, 914],
    "car": [407, 436, 468, 511, 609, 627, 656, 661, 751, 817],
    "cat": [281, 282, 283, 284, 285, 286, 287],
    "edible_fruit": [948, 949, 950, 951, 952, 953, 954, 955, 956, 957],
    "fungus": [991, 993, 994, 995, 996, 997],
    "truck": [ 555, 569, 656, 675, 717, 734, 864, 867],
}
# fmt: on


@register_dataset("imagenet")
class ImageNet(DatasetConfiguration):
    selected_classes = list(range(1000))

    def __init__(self):
        # remark: we need to set this manually.
        self.num_classes = 1000

        self._normalizer = transforms.Normalize(
            # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L44
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # ref: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_presets.py#L38
        self.input_transformation = DEFAULT_TRANSFORMATION

        # ref: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L238
        self.input_training_transformation = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalizer,
            ]
        )

        np.testing.assert_allclose(
            self.input_transformation.mean, self._normalizer.mean
        )

        self.dataclass = tvd.ImageNet
        self.root = DATADIR / "imagenet"

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:
        return self.dataclass(
            root=self.root,
            split="train" if train_split else "val",
            transform=self.input_transformation,
            target_transform=target_transform,
        )

    def transform_target(self, target: int) -> int:
        return target
