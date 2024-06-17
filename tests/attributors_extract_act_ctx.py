import pytest

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from xaikd import attributors, datasets, utils, models

DEVICE = utils.get_device()

NUMBER_OF_SMALL_DATASET = 7
NUMBER_OF_SPATIAL_LOCATIONS = 8


class CIFAR100VerySmall(datasets.cifar100.CIFAR100):
    def loader(self, batch_size=64, num_workers=2, train_split=False):
        ds = self.create_subset(train_split=train_split)

        np.random.seed(1)

        return DataLoader(
            Subset(
                ds, np.random.permutation(ds.data.shape[0])[:NUMBER_OF_SMALL_DATASET]
            ),
            num_workers=num_workers,
            batch_size=batch_size,
        )


class ImageNetVerySmall(datasets.imagenet.ImageNet):
    def loader(self, batch_size=64, num_workers=2, train_split=False):

        trng = torch.Generator()
        trng.manual_seed(1)
        x = torch.randn((NUMBER_OF_SMALL_DATASET, 3, 224, 224), generator=trng)
        y = torch.randint(
            low=0, high=1000, size=(NUMBER_OF_SMALL_DATASET,), generator=trng
        )
        ds = TensorDataset(x, y)

        return DataLoader(
            ds,
            num_workers=num_workers,
            batch_size=batch_size,
        )


def _test_extract_activation_context(model_name, dataset_class, layer):
    model = models.get_trained_model(model_name).to(DEVICE)

    dataset = dataset_class()

    output_dims = models.get_layer_output_dimensions(model, layer)

    train_dl = dataset.loader(train_split=True)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer=layer,
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=attributors.WinningClassEvidence(
            num_classes=dataset.num_classes
        ),
        rng=np.random.default_rng(seed=1),
    )

    assert arr_act.shape == (
        NUMBER_OF_SPATIAL_LOCATIONS * NUMBER_OF_SMALL_DATASET,
        output_dims,
    )

    assert arr_act.shape == arr_ctx.shape


@pytest.mark.gpu()
@pytest.mark.parametrize("layer", ["layer1", "layer2", "layer3", "layer4"])
def test_extract_activation_context_cifar100(layer):
    _test_extract_activation_context("cifar100-resnet18-v1", CIFAR100VerySmall, layer)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "model_name,layer",
    [
        (
            "imagenet-resnet18-tv",
            "layer1",
        ),
        (
            "imagenet-resnet18-tv",
            "layer2",
        ),
        (
            "imagenet-vgg16-tv",
            "features.23",
        ),
        (
            "imagenet-vitb-tv",
            "encoder.layers.8",
        ),
    ],
)
def test_extract_activation_context_imagenet(model_name, layer):
    _test_extract_activation_context(model_name, ImageNetVerySmall, layer)


@pytest.mark.parametrize("seed", [1, 2])
def test_extract_activation_context_with_same_seed_different_run(seed):
    model = models.get_trained_model("cifar100-resnet18-v1").to(DEVICE)

    dataset = CIFAR100VerySmall()

    train_dl = dataset.loader(train_split=True)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer="layer3",
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=attributors.TargetClassEvidence(num_classes=dataset.num_classes),
        rng=np.random.default_rng(seed=1),
    )

    expected_arr_act, expected_arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer="layer3",
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=attributors.TargetClassEvidence(num_classes=dataset.num_classes),
        rng=np.random.default_rng(seed=1),
    )

    np.testing.assert_allclose(arr_act, expected_arr_act, atol=1e-6)

    np.testing.assert_allclose(arr_ctx, expected_arr_ctx, atol=1e-6)
