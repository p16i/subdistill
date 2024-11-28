from torchvision.datasets import MNIST, FashionMNIST, VisionDataset

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
