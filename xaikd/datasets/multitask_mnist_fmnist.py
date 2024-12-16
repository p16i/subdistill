from torchvision.datasets import MNIST, FashionMNIST, VisionDataset, EMNIST
from torchvision.transforms import functional as TF


import numpy as np

from PIL import Image

from pathlib import Path

from typing import Union, Optional, Callable, Tuple, Any


class MultiTaskMNISTFashionMNIST(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        seed=1,
    ) -> None:
        rng = np.random.default_rng(seed=seed)
        ds_mnist = MNIST(train=train, root=root)
        ds_fmnist = FashionMNIST(train=train, root=root)

        assert len(ds_mnist) == len(ds_fmnist)
        num_samples = len(ds_mnist)

        selected_mnist_indices = rng.permutation(num_samples)
        selected_fmnist_indices = rng.permutation(num_samples)

        arr_mnist_first = rng.binomial(1, 0.5, size=(num_samples))

        arr_task_locations = []
        arr_data = []
        arr_targets = []
        # concatenate the two datasets
        for is_mnist_first, six_mnist, six_fmnist in zip(
            arr_mnist_first, selected_mnist_indices, selected_fmnist_indices
        ):
            x_mnist, y_mnist = ds_mnist[six_mnist]
            x_mnist = np.array(x_mnist)

            x_fmnist, y_fmnist = ds_fmnist[six_fmnist]
            x_fmnist = np.array(x_fmnist)

            if is_mnist_first:
                x = np.hstack([x_mnist, x_fmnist])
                task_locs = (0, 1)  # task 0 and 1
            else:
                x = np.hstack([x_fmnist, x_mnist])
                task_locs = (1, 0)  # task 1 and 0

            x = Image.fromarray(x, mode="L")
            y = (y_mnist, y_fmnist)

            arr_targets.append(y)
            arr_data.append(x)
            arr_task_locations.append(task_locs)

        self.data = arr_data
        self.targets = np.array(arr_targets).astype(int)

        self.arr_task_locations = np.array(arr_task_locations)

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, task_targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, task_targets

    def __len__(self) -> int:
        return self.targets.shape[0]


class MultiTaskEMNISTFashionMNIST(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        seed=1,
    ) -> None:
        rng = np.random.default_rng(seed=seed)

        ds_emnist = EMNIST(split="balanced", train=train, download=True, root=root)
        ds_fmnist = FashionMNIST(train=train, root=root)

        num_samples = len(ds_fmnist)

        selected_emnist_indices = rng.permutation(num_samples)
        selected_fmnist_indices = rng.permutation(num_samples)

        arr_emnist_first = rng.binomial(1, 0.5, size=(num_samples))

        arr_task_locations = []
        arr_data = []
        arr_targets = []
        # concatenate the two datasets
        for is_emnist_first, six_emnist, six_fmnist in zip(
            arr_emnist_first, selected_emnist_indices, selected_fmnist_indices
        ):
            x_emnist, y_emnist = ds_emnist[six_emnist]

            # Remark: EMNIST comes with an uninituive orientation.
            # See alos https://github.com/tensorflow/datasets/commit/76553955ca0d56a50170374c3e28f1a4b9720601
            # The code below is from https://stackoverflow.com/a/54513835.
            x_emnist = TF.rotate(x_emnist, -90)
            x_emnist = TF.hflip(x_emnist)

            x_emnist = np.array(x_emnist)

            x_fmnist, y_fmnist = ds_fmnist[six_fmnist]
            x_fmnist = np.array(x_fmnist)

            if is_emnist_first:
                x = np.hstack([x_emnist, x_fmnist])
                task_locs = (0, 1)  # task 0 and 1
            else:
                x = np.hstack([x_fmnist, x_emnist])
                task_locs = (1, 0)  # task 1 and 0

            x = Image.fromarray(x, mode="L")
            y = (y_emnist, y_fmnist)

            arr_targets.append(y)
            arr_data.append(x)
            arr_task_locations.append(task_locs)

        self.data = arr_data
        self.targets = np.array(arr_targets).astype(int)

        self.arr_task_locations = np.array(arr_task_locations)

        self.transform = transform
        self.num_classes = [
            len(ds_emnist.classes),
            len(ds_fmnist.classes),
        ]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, task_targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, task_targets

    def __len__(self) -> int:
        return self.targets.shape[0]
