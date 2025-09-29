import typing

import numpy as np
from numpy import typing as npt
from torchvision import transforms
from torchvision import datasets as tvd

from .. import (
    TORCHVISION_DATASET_DOWNLOAD,
    DATADIR,
    DatasetConfiguration,
)
from ..register import register_dataset


from xaikd import utils


class TorchVisionCIFAR100CWithSeverity(tvd.CIFAR100):
    def __init__(
        self,
        *args,
        x: npt.NDArray,
        y: list[int],
        arr_sample_severity: list[int],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.arr_sample_severity = arr_sample_severity
        self.data = x
        self.targets = y

        assert len(self.data) == len(self.arr_sample_severity) == len(self.targets)


class CIFAR100CorruptionBase(DatasetConfiguration):
    @property
    def _normalizer(self):
        # ref: https://github.com/RobustBench/robustbench/blob/78fcc9e48a07a861268f295a777b975f25155964/robustbench/model_zoo/cifar100.py#L238
        return transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
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
        return TorchVisionCIFAR100CWithSeverity

    def create_subset(
        self,
        train_split=False,
    ) -> tvd.CIFAR100:
        if train_split:
            ds_original = tvd.CIFAR100(
                root=str(DATADIR / "cifar100"),
                train=train_split,
                download=TORCHVISION_DATASET_DOWNLOAD,
            )

            arr_sample_severity = [0] * len(ds_original.targets)

            return TorchVisionCIFAR100CWithSeverity(
                root=str(DATADIR / "cifar100"),
                x=ds_original.data,
                y=ds_original.targets,
                arr_sample_severity=arr_sample_severity,  # type: ignore
                target_transform=self.target_transform,
                transform=self.input_transformation,
            )
        else:
            (ds_test, _) = self._create_val_test_split()
            return ds_test

    def _create_val_test_split(self):
        num_severity = 5
        n_examples = 12500
        test_val_ratio = 0.8
        n_test_examples = int(n_examples * test_val_ratio)

        arr_test_x = []
        arr_test_y = []
        arr_test_severity = []

        arr_val_x = []
        arr_val_y = []
        arr_val_severity = []

        rng = np.random.default_rng(seed=1)

        for severity in range(1, num_severity + 1):
            x, y = utils.robustbench.load_cifar100c(
                n_examples=n_examples,
                severity=severity,
                data_dir=str(DATADIR / "cifar100c"),
            )

            index = rng.permutation(x.shape[0])

            test_index = index[:n_test_examples]
            np.testing.assert_allclose(len(test_index), n_test_examples)

            val_index = index[n_test_examples:]

            arr_test_x.append(x[test_index, ...])
            arr_test_y.append(y[test_index, ...])
            arr_test_severity.extend([severity] * len(test_index))

            arr_val_x.append(x[val_index, ...])
            arr_val_y.append(y[val_index, ...])
            arr_val_severity.extend([severity] * len(val_index))

        arr_ds = []

        for _x, _y, _severity in [
            (arr_test_x, arr_test_y, arr_test_severity),
            (arr_val_x, arr_val_y, arr_val_severity),
        ]:
            _x = np.concatenate(_x, axis=0)
            _y = np.concatenate(_y, axis=0).tolist()

            assert _x.shape[0] == len(_y) == len(_severity)

            ds = TorchVisionCIFAR100CWithSeverity(
                root=str(DATADIR / "cifar100"),
                x=_x,
                y=_y,
                arr_sample_severity=_severity,
                train=False,
                transform=self.input_transformation,
                download=TORCHVISION_DATASET_DOWNLOAD,
                target_transform=self.target_transform,
            )

            assert len(ds.data) == len(ds.arr_sample_severity) == len(ds.targets)

            arr_ds.append(ds)

        return arr_ds


@register_dataset("cifar100c")
class CIFAR100C(CIFAR100CorruptionBase):
    @property
    def selected_classes(self):
        return list(range(100))

    @property
    def num_classes(self) -> int:
        return 100

    @property
    def target_transform(self) -> typing.Union[None, typing.Callable]:
        return None
