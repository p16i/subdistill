import pytest

import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from xaikd import attributors, datasets, utils, models, logit_modifiers

DEVICE = utils.get_device()

NUMBER_OF_SMALL_DATASET = 7
NUMBER_OF_SPATIAL_LOCATIONS = 8


class CIFAR100VerySmall(datasets.cifar100.original.CIFAR100):
    def loader(self, batch_size=64, num_workers=2, train_split=False):
        ds = self.create_subset(train_split=train_split)

        np.random.seed(1)

        return DataLoader(
            Subset(
                ds,
                np.random.permutation(ds.data.shape[0])[
                    :NUMBER_OF_SMALL_DATASET
                ].tolist(),
            ),
            num_workers=num_workers,
            batch_size=batch_size,
        )


class ImageNetVerySmall(datasets.imagenet.original.ImageNet):
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
    model = models.get_trained_model(model_name)

    dataset = dataset_class()
    train_dl = dataset.loader(train_split=True)
    output_dims = utils.get_dimensions_at_layers(model, train_dl, [layer])[layer]

    model = model.to(DEVICE)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer=layer,
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=logit_modifiers.MultiClassWinningLogit(),
        strict_mode=True,
        rng=np.random.default_rng(seed=1),
    )

    assert arr_act.shape == (
        NUMBER_OF_SMALL_DATASET,
        output_dims,
        NUMBER_OF_SPATIAL_LOCATIONS,
    )

    assert arr_act.shape == arr_ctx.shape


@pytest.mark.gpu()
@pytest.mark.parametrize("layer", ["layer1", "layer2", "layer3", "layer4"])
def test_extract_activation_context_cifar100(layer):
    _test_extract_activation_context("cifar100-resnet18-v1", CIFAR100VerySmall, layer)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "model_name,arr_layers",
    [
        ("imagenet-resnet18-tv", "layer3,layer4"),
        ("imagenet-resnet50-tv", "layer3,layer4"),
        ("imagenet-vgg16-tv", "features.23,features.30"),
        ("imagenet-nfnetf0-dm", "stages.2,stages.3"),
        ("imagenet-vitb-tv", "encoder.layers.8,encoder.layers.11"),
    ],
)
def test_extract_activation_context_imagenet(model_name, arr_layers):
    for layer in arr_layers.split(","):
        _test_extract_activation_context(model_name, ImageNetVerySmall, layer)


@pytest.mark.parametrize("seed", [1, 2])
def test_extract_activation_context_with_same_seed_different_run(seed):
    model = models.get_trained_model("cifar100-resnet18-v1").to(DEVICE)

    dataset = CIFAR100VerySmall()

    train_dl = dataset.loader(train_split=True)

    # todo: this should test extract_act_extract_grad
    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer="layer3",
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=logit_modifiers.MultiClassTargetLogit(),
        rng=np.random.default_rng(seed=seed),
    )

    expected_arr_act, expected_arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer="layer3",
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=logit_modifiers.MultiClassTargetLogit(),
        rng=np.random.default_rng(seed=seed),
    )

    np.testing.assert_allclose(arr_act, expected_arr_act, atol=1e-6)

    np.testing.assert_allclose(arr_ctx, expected_arr_ctx, atol=1e-6)


@pytest.mark.parametrize(
    "num_data_points,batch_size",
    [
        (10, 10),
        (10, 5),
        (100, 5),
        (100, 20),
    ],
)
def test_extract_activation_grad(num_data_points, batch_size):
    seed = 1
    torch.manual_seed(seed)

    X = torch.randn(num_data_points, 3, 32, 32)
    y = torch.randint(0, 10, size=(num_data_points,))

    model_part1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.GELU(),
        nn.Conv2d(16, 26, kernel_size=3),
        nn.GELU(),
    )

    model_part2 = nn.Sequential(
        nn.Conv2d(26, 32, kernel_size=3),
        nn.GELU(),
        nn.Conv2d(32, 7, kernel_size=3),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(start_dim=1),
        nn.Linear(7, 5),
    )

    logit_modifier = logit_modifiers.MultiClassDifferenceTop2Logits()

    act: torch.Tensor = model_part1(X).detach()
    act.requires_grad_(True)
    arr_expected_logodd = model_part2(act)
    logit_modifier(arr_expected_logodd, None).sum().backward()

    grad = act.grad
    assert grad is not None

    num_spatial_locations = int(np.prod(act.shape[2:]))

    arr_expected_acts, arr_expected_grads = utils.subsample_tensors(
        act.detach().numpy(),
        grad.detach().numpy(),
        num_locations=num_spatial_locations,
        rng=np.random.default_rng(seed=seed),
    )
    expected_mean = utils.flatten_3d_tensor(arr_expected_acts).mean(axis=0)
    arr_expected_acts -= expected_mean[None, :, None]

    model = nn.Sequential(
        OrderedDict([("layer1", model_part1), ("layer2", model_part2)])
    )

    dl = DataLoader(
        TensorDataset(X, y),
        shuffle=False,
        batch_size=batch_size,
    )

    (
        _,
        arr_actual_acts,
        arr_actual_grads,
        mean_act,
    ) = attributors.extract_activation_grad(
        model=model,
        layer="layer1",
        dataloader=dl,
        logit_modifier=logit_modifier,
        device="cpu",
        rng=np.random.default_rng(seed=seed),
        number_of_selected_spatial_locations=num_spatial_locations,
    )

    np.testing.assert_allclose(mean_act, expected_mean)
    np.testing.assert_allclose(
        arr_actual_acts,
        arr_expected_acts,
    )

    np.testing.assert_allclose(
        arr_actual_grads,
        arr_expected_grads,
    )

    # we don't have this anymore.k
    # np.testing.assert_allclose(
    #     arr_actual_logodd,
    #     arr_expected_logodd.detach().cpu().numpy(),
    # )
