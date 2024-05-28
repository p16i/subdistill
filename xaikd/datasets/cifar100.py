import typing

from dataclasses import dataclass
from functools import partial

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision import datasets as tvd

from xaikd import constants


from . import (
    TORCHVISION_DATASET_DOWNLOAD,
    DATADIR,
    DATASETS,
    register_dataset,
    DatasetConfiguration,
)


@register_dataset("cifar100")
@dataclass
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

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        return target

    def create_subset(
        self,
        train_split=False,
        target_transform: typing.Union[None, typing.Callable] = None,
    ) -> Dataset:
        return self.dataclass(
            root=self.root,
            train=train_split,
            transform=self.input_transformation,
            download=TORCHVISION_DATASET_DOWNLOAD,
            target_transform=target_transform,
        )


@dataclass
class Cifar100SuperClassesDataset(DatasetConfiguration):
    def __init__(
        self,
        super_class: str,
        num_train_samples: typing.Union[None, int] = None,
        verbose=False,
    ):
        self.base = CIFAR100()
        df_meta = pd.read_csv(constants.CIFAR100_SUPER_CLASS_MAPPING)

        df_selected = df_meta[df_meta.coarse_label_name == super_class]
        df_selected = df_selected.sort_values(by="fine_label")
        if verbose:
            print(
                f"We are building `cifar100-{super_class}` containing {df_selected.shape[0]} fine classes"
            )
            for row in df_selected.to_dict("records"):
                print("> %s (%d)" % (row["fine_label_name"], row["fine_label"]))

        # remark: the targets are defined in the CIFAR100 dataset.
        self.selected_classes = df_selected.fine_label.values.tolist()

        self.num_classes = len(self.selected_classes)

        self._normalizer = self.base._normalizer
        self.input_transformation = self.base.input_transformation
        self.input_training_transformation = self.base.input_training_transformation

        # change name to mapping_old_and_new_target_indices
        # converting from old target (original dataset) to new target {0, 1,...})
        self._target_mapping = dict(
            zip(self.selected_classes, range(len(self.selected_classes)))
        )

    def create_subset(self, train_split=False) -> Dataset:
        ds = self.base.create_subset(
            train_split=train_split, target_transform=lambda t: self._target_mapping[t]
        )
        labels = ds.targets

        selected_data_indices = np.argwhere(
            np.isin(labels, self.selected_classes)
        ).reshape(-1)

        # here, we select samples belong to those targets.
        ds.data = ds.data[selected_data_indices, :]

        targets = np.array(ds.targets)[selected_data_indices].tolist()
        assert np.isin(targets, self.selected_classes).all()

        # remark: the targets here are still in the old system.
        # They will be converted to the new zero-indexing with target_transforms.
        # todo: add test
        #   comparing naive cifar100 and this dataset should have the same val
        ds.targets = targets

        return ds


for super_class in constants.CIFAR100_SUPER_CLASSES:
    DATASETS[f"cifar100-{super_class}"] = partial(
        Cifar100SuperClassesDataset, super_class=super_class
    )
