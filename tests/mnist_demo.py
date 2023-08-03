import pytest
import numpy as np
from torch.utils.data import DataLoader

import xaikd.mnist_demo as mnist_demo


@pytest.mark.skip("to fix")
def test_subclass_selection():
    considered_classes = (4, 9)
    total_samples_per_class = 500
    train_subset, val_subset = mnist_demo.build_subclasses_loader(
        considered_classes, total_samples_per_class
    )

    actual_total_samples = 0
    for x, y in DataLoader(train_subset, num_workers=2, batch_size=128):
        assert np.isin(y.numpy(), considered_classes).all()
        actual_total_samples += x.shape[0]

    for x, y in DataLoader(val_subset, num_workers=2, batch_size=128):
        assert np.isin(y.numpy(), considered_classes).all()

    assert actual_total_samples == total_samples_per_class * len(considered_classes)
