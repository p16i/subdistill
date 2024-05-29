import typing

from dataclasses import dataclass
from functools import partial

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, random_split

from torchvision import transforms
from torchvision import datasets as tvd

from PIL import Image, ImageDraw


from xaikd import constants


from . import (
    TORCHVISION_DATASET_DOWNLOAD,
    DATADIR,
    DATASETS,
    register_dataset,
    DatasetConfiguration,
)

CLEVER_HAN_SYMBOL = "+"
COLOR = "red"


def add_cleverhan_symbol(img, rng: np.random.Generator):
    copied_img = img.copy()

    x = rng.integers(low=0, high=31 - 4)
    # remark: because the anchor attribute doesn't seem to work with the default font,
    # we therefoe adjust by - 3 manually here to compensate the empty space above `+` from the default font.
    # cf. ./notebooks/2023-10-s16/dev-add-symbol-to-img.ipynb
    y = rng.integers(low=0 - 3, high=31 - 4 - 3)

    location = (x, y)

    ImageDraw.Draw(copied_img).text(
        location,
        text=CLEVER_HAN_SYMBOL,
        fill=COLOR,
    )
    return copied_img


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
class Cifar100SuperClassesDataset(CIFAR100):
    def __init__(
        self,
        super_class: str,
        verbose=False,
    ):
        super().__init__()

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

        # change name to mapping_old_and_new_target_indices
        # converting from old target (original dataset) to new target {0, 1,...})
        self._target_mapping = dict(
            zip(self.selected_classes, range(len(self.selected_classes)))
        )

    def create_subset(self, train_split=False) -> Dataset:
        ds = super().create_subset(
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


@dataclass
class Cifar100SuperClassesWithSpuriousFeatureDataset(Cifar100SuperClassesDataset):
    seed = 1

    def __init__(
        self,
        super_class: str,
        contamination_level: float,
    ):
        super().__init__(super_class=super_class)

        self.contamination_level = contamination_level

    def create_subset(self, train_split=False) -> Dataset:

        ds = super().create_subset(train_split=train_split)

        rng = np.random.default_rng(seed=self.seed)

        if train_split:
            victim_class = np.min(self.selected_classes)
            # for `training` set,  we are only interested in only a class
            total_possible_victims = (np.array(ds.targets) == victim_class).sum()
            potential_victim_indices = (
                np.argwhere(np.array(ds.targets) == victim_class).reshape(-1).tolist()
            )
        else:
            # for `validation` set,  samples from all classes have the same
            # likelihood of having the spurious feature.
            total_possible_victims = len(ds.targets)
            potential_victim_indices = list(range(total_possible_victims))

        total_datapoint_with_spurious = int(
            np.floor(len(potential_victim_indices) * self.contamination_level)
        )

        victim_datapoint_indices = rng.permutation(potential_victim_indices)[
            :total_datapoint_with_spurious
        ]

        # for testing purpose
        ds.victim_indices = victim_datapoint_indices

        print(
            f"> {len(victim_datapoint_indices)} victims (total={total_datapoint_with_spurious})"
        )

        for ix in victim_datapoint_indices:
            img = Image.fromarray(ds.data[ix])

            new_img = add_cleverhan_symbol(img, rng)

            ds.data[ix] = np.array(new_img)

        return ds


class Cifar100ValSplitSuperClassesWithSpuriousFeatureDataset(
    Cifar100SuperClassesDataset
):
    seed = 1

    def __init__(
        self,
        super_class: str,
        contamination_level: float,
    ):
        super().__init__(super_class=super_class)

        self.contamination_level = contamination_level

    def create_subset(self, train_split=False) -> Dataset:
        trng = torch.Generator()
        trng.manual_seed(self.seed)

        ds = super().create_subset(train_split=True)
        total_size = ds.data.shape[0]
        np.testing.assert_allclose(total_size, 500 * len(self.selected_classes))
        subsets = random_split(
            # if `use-val-split=True, both training and testing sets
            # come from the training set.
            ds,
            [0.8, 0.2],
            generator=trng,
        )

        rng = np.random.default_rng(seed=self.seed)

        selected_subset = subsets[0] if train_split else subsets[1]

        subset_data_indices = selected_subset.indices

        # here, we override the original data
        ds.data = ds.data[subset_data_indices]
        ds.targets = np.array(ds.targets)[subset_data_indices].tolist()

        targets = ds.targets

        print(f"[train_split={train_split}] len(indices):={ds.data.shape[0]} ")

        np.testing.assert_allclose(
            ds.data.shape[0],
            total_size
            * (
                constants.TRAINING_VAL_SPLIT_RATIO
                if train_split
                else 1 - constants.TRAINING_VAL_SPLIT_RATIO
            ),
        )

        if train_split:
            victim_class = np.min(self.selected_classes)

            potential_victim_indices = (
                np.argwhere(targets == victim_class).reshape(-1).tolist()
            )
            total_possible_victims = len(potential_victim_indices)

            expected_total = 500 * constants.TRAINING_VAL_SPLIT_RATIO

            np.testing.assert_allclose(total_possible_victims, expected_total, atol=10)
        else:
            # for `validation` set,  samples from all classes have the same
            # likelihood of having the spurious feature.
            total_possible_victims = ds.data.shape[0]
            potential_victim_indices = np.arange(total_possible_victims)

            np.testing.assert_allclose(
                total_possible_victims,
                500 * 5 * (1 - constants.TRAINING_VAL_SPLIT_RATIO),
            )

        total_datapoint_with_spurious = int(
            np.floor(len(potential_victim_indices) * self.contamination_level)
        )

        victim_datapoint_indices = rng.permutation(potential_victim_indices)[
            :total_datapoint_with_spurious
        ]

        # for testing purpose
        ds.victim_indices = victim_datapoint_indices

        print(
            f"> {len(victim_datapoint_indices)} victims (total={total_datapoint_with_spurious})"
        )

        for ix in victim_datapoint_indices:
            img = Image.fromarray(ds.data[ix])

            new_img = add_cleverhan_symbol(img, rng)

            ds.data[ix] = np.array(new_img)

        return ds


def ano():
    for super_class in constants.CIFAR100_SUPER_CLASSES:
        slug = f"cifar100-{super_class}"
        DATASETS[slug] = partial(Cifar100SuperClassesDataset, super_class=super_class)

        for lvl in [0.125, 0.25, 0.5, 1.0]:
            sslug = "--".join([slug, "spurious-plussign", str(lvl)])

            DATASETS[sslug] = partial(
                Cifar100SuperClassesWithSpuriousFeatureDataset,
                super_class=super_class,
                contamination_level=lvl,
            )

    for lvl in [0.0, 0.5, 1.0]:
        DATASETS[
            "--".join(["cifar100-valsplit-people", "spurious-plussign", str(lvl)])
        ] = partial(
            Cifar100ValSplitSuperClassesWithSpuriousFeatureDataset,
            super_class="people",
            contamination_level=lvl,
        )


ano()
