import typing

import numpy as np
from numpy import typing as npt
from torchvision import transforms
from torchvision import datasets as tvd
from functools import partial

from .. import (
    TORCHVISION_DATASET_DOWNLOAD,
    DATADIR,
    DatasetConfiguration,
)
from ..register import add_dataset_to_registry, register_dataset
from ..cifar100 import get_fineclass_names_indices_of_superclass


from xaikd import utils, constants


# fixme: used it in subclass
def _cifar100_select_only_classes_within_selected_classes(
    selected_classes: list[int], ds: tvd.CIFAR100
) -> tvd.CIFAR100:
    labels = ds.targets

    selected_data_indices = np.argwhere(np.isin(labels, selected_classes)).reshape(-1)

    # here, we select samples belong to those targets.
    ds.data = ds.data[selected_data_indices, :]

    targets = np.array(ds.targets)[selected_data_indices].tolist()
    assert np.isin(targets, selected_classes).all()

    # remark: the targets here are still in the old system.
    # They will be converted to the new zero-indexing with target_transforms.
    ds.targets = targets

    return ds


class TorchVisionCIFAR100CWithSeverity(tvd.CIFAR100):
    def __init__(
        self,
        *args,
        x: npt.NDArray,
        y: list[int],
        arr_sample_severity: list[int],
        **kwargs,
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

    def _create_val_test_split(self) -> tuple[tvd.CIFAR100, tvd.CIFAR100]:
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

            x = x.permute(0, 2, 3, 1).numpy()  # to HWC and numpy
            x = (x * 255).astype(np.uint8)  # to [0, 255] uint8

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

        return tuple(arr_ds)


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


class CIFAR100CSuperclass(CIFAR100C):
    def __init__(
        self,
        superclass: str,
        verbose=False,
    ):
        super().__init__()

        self._superclass = superclass

        (
            arr_fineclass_names,
            arr_fineclass_idx,
        ) = get_fineclass_names_indices_of_superclass(superclass)

        if verbose:
            print(
                f"We are building `cifar100-{superclass}` containing {len(arr_fineclass_names)} fine classes"
            )
            for idx, name in zip(arr_fineclass_idx, arr_fineclass_names):
                print(f"> {name} ({idx})")

        # remark: the targets are defined in the CIFAR100 dataset.
        self._selected_classes = arr_fineclass_idx

        # change name to mapping_old_and_new_target_indices
        # converting from old target (original dataset) to new target {0, 1,...})
        self._target_mapping = dict(
            zip(self.selected_classes, range(len(self.selected_classes)))
        )

    @property
    def selected_classes(self):
        return self._selected_classes

    @property
    def num_classes(self) -> int:
        return len(self._selected_classes)

    @property
    def target_transform(self):
        def _transform(t):
            return self._target_mapping[t]

        return _transform

    def create_subset(self, train_split=False) -> tvd.CIFAR100:
        if train_split:
            ds = super().create_subset(train_split)
            return _cifar100_select_only_classes_within_selected_classes(
                selected_classes=self.selected_classes,
                ds=ds,
            )
        else:
            return super().create_subset(train_split)

    def _create_val_test_split(self) -> tuple[tvd.CIFAR100, tvd.CIFAR100]:
        ds_test, ds_val = super()._create_val_test_split()

        ds_test = _cifar100_select_only_classes_within_selected_classes(
            selected_classes=self.selected_classes,
            ds=ds_test,
        )

        ds_val = _cifar100_select_only_classes_within_selected_classes(
            selected_classes=self.selected_classes,
            ds=ds_val,
        )

        return ds_test, ds_val


def construct_variant_datasets():
    for super_class in constants.CIFAR100_SUPER_CLASSES:
        slug = f"cifar100c-{super_class}"

        add_dataset_to_registry(
            slug, partial(CIFAR100CSuperclass, superclass=super_class)
        )


construct_variant_datasets()
