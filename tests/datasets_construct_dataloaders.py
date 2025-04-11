import pytest

import numpy as np

from torchvision.transforms import (
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Normalize,
)


from torchvision.models import ResNet18_Weights

from xaikd import datasets, metrics, constants


@pytest.mark.parametrize("training_data_ratio", [0.01, 0.5, 1.0])
def test_ds_val_set_same_as_ds_test_when_use_validation_set_disabled(
    training_data_ratio,
):
    use_validation_set = False

    dataset = datasets.construct("cifar100")

    _, _, dl_val, dl_test = datasets.construct_dataloaders(
        dataset=dataset,
        training_data_ratio=training_data_ratio,
        seed=1,
        use_validation_set=use_validation_set,
        training_batch_size=16
    )

    np.testing.assert_allclose(
        next(iter(dl_val))[0],
        next(iter(dl_test))[0],
    )


def test_train_transform_cifar100():
    use_validation_set = False

    dataset = datasets.construct("cifar100")

    dl_train, dl_train_with_aug, _, _ = datasets.construct_dataloaders(
        dataset=dataset,
        training_data_ratio=1.0,
        seed=1,
        use_validation_set=use_validation_set,
        training_batch_size=16
    )

    for actual, expected_class in zip(
        dl_train.dataset.dataset.transform.transforms, [ToTensor, Normalize]  # type: ignore
    ):
        assert isinstance(actual, expected_class)

    for actual, expected_class in zip(
        dl_train_with_aug.dataset.dataset.transform.transforms,  # type: ignore
        [RandomCrop, RandomHorizontalFlip],
    ):
        assert isinstance(actual, expected_class)


@pytest.mark.slow
def test_train_transform_imagenet():
    use_validation_set = False

    dataset = datasets.construct("imagenet")

    dl_train, dl_train_with_aug, _, _ = datasets.construct_dataloaders(
        dataset=dataset,
        training_data_ratio=1.0,
        seed=1,
        use_validation_set=use_validation_set,
        training_batch_size=16
    )

    assert isinstance(
        dl_train.dataset.dataset.transform,  # type: ignore
        type(ResNet18_Weights.IMAGENET1K_V1.transforms()),
    )

    for actual, expected_class in zip(
        dl_train_with_aug.dataset.dataset.transform.transforms,  # type: ignore
        [RandomResizedCrop, RandomHorizontalFlip],
    ):
        assert isinstance(actual, expected_class)


@pytest.mark.parametrize("training_data_ratio", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("use_validation_set", [False, True])
def test_data_size_when_val_split(training_data_ratio, use_validation_set):
    dataset_name = "cifar100"
    dataset = datasets.construct(dataset_name)

    total_original_train = len(dataset.create_subset(train_split=True))
    total_original_test = len(dataset.create_subset(train_split=False))

    dl_train, dl_train_with_aug, dl_val, dl_test = datasets.construct_dataloaders(
        dataset,
        training_data_ratio=training_data_ratio,
        seed=1,
        use_validation_set=use_validation_set,
        training_batch_size=16
    )

    np.testing.assert_allclose(len(dl_train.dataset), len(dl_train_with_aug.dataset))  # type: ignore
    np.testing.assert_allclose(len(dl_test.dataset), total_original_test)  # type: ignore

    if use_validation_set and training_data_ratio > constants.TRAINING_VAL_SPLIT_RATIO:
        expected_training_ratio = constants.TRAINING_VAL_SPLIT_RATIO
    else:
        expected_training_ratio = training_data_ratio

    np.testing.assert_allclose(
        len(dl_train.dataset) / total_original_train,  # type: ignore
        expected_training_ratio,
        atol=1e-3,
    )

    if use_validation_set:
        np.testing.assert_allclose(
            len(dl_val.dataset) / total_original_train,  # type: ignore
            1 - constants.TRAINING_VAL_SPLIT_RATIO,
            atol=1e-3,
        )

    else:
        np.testing.assert_allclose(len(dl_val.dataset), len(dl_test.dataset))  # type: ignore
