import pytest
import torch
import numpy as np

from xaikd import attributors, models
from xaikd import utils, datasets

from torch.utils.data import DataLoader, Subset

DEVICE = utils.get_device()

NUMBER_OF_SMALL_DATASET = 7
NUMBER_OF_SPATIAL_LOCATIONS = 8


class CIFAR100VerySmall(datasets.CIFAR100):
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


@pytest.mark.gpu()
@pytest.mark.parametrize("layer", ["layer1", "layer2", "layer3", "layer4"])
def test_extract_activation_context(layer):
    model = models.get_model("cifar100-resnet18-p1").to(DEVICE)

    dataset = CIFAR100VerySmall()

    output_dims = models.get_layer_output_dimensions(model, layer)

    train_dl = dataset.loader(train_split=True)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer=layer,
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=attributors.OneClassEvidence(num_classes=dataset.num_classes),
        rng=np.random.default_rng(seed=1),
    )

    assert arr_act.shape == (
        NUMBER_OF_SPATIAL_LOCATIONS * NUMBER_OF_SMALL_DATASET,
        output_dims,
    )

    assert arr_act.shape == arr_ctx.shape


@pytest.mark.parametrize("seed", [1, 2])
def test_extract_activation_context_with_same_seed_different_run(seed):
    model = models.get_model("cifar100-resnet18-p1").to(DEVICE)

    dataset = CIFAR100VerySmall()

    train_dl = dataset.loader(train_split=True)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer="layer3",
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=attributors.OneClassEvidence(num_classes=dataset.num_classes),
        rng=np.random.default_rng(seed=1),
    )

    expected_arr_act, expected_arr_ctx = attributors.extract_activation_context(
        model=model,
        data_loader=train_dl,
        dataset=dataset,
        layer="layer3",
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=attributors.OneClassEvidence(num_classes=dataset.num_classes),
        rng=np.random.default_rng(seed=1),
    )

    np.testing.assert_allclose(arr_act, expected_arr_act, atol=1e-6)

    np.testing.assert_allclose(arr_ctx, expected_arr_ctx, atol=1e-6)


def test_logit_modifier_oneclass():
    dataset = datasets.construct("cifar10")

    all_classes = set(range(dataset.num_classes))
    class1 = 1

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))

    logits_mod_single = attributors.OneClassEvidence(num_classes=dataset.num_classes)(
        logits, torch.tensor([class1] * 2)
    )

    assert (logits_mod_single[:, class1] == logits[:, class1]).all()
    assert (logits_mod_single[:, list(all_classes.difference([class1]))] == 0).all()


@pytest.mark.parametrize("target", ("abc", None))
def test_logit_modifier_logodd(target):
    dataset: datasets.TwoClassesDataset = datasets.construct("cifar10-1vs8")

    all_classes = set(range(dataset.num_classes))
    class1, class2 = dataset.selected_classes

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))

    logits_mod_logood = attributors.LogOddEvidence((class1, class2))(logits, target)

    assert (logits_mod_logood[:, class1] == logits[:, class1]).all()
    assert (logits_mod_logood[:, class2] == -logits[:, class2]).all()
    assert (
        logits_mod_logood[:, list(all_classes.difference([class1, class2]))] == 0
    ).all()


def test_logit_modifier_selected_classes():
    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        "cifar100-people"
    )

    # selected classes in the new label index system.
    selected_classes = np.array([0, 1])

    assert (selected_classes <= (dataset.num_classes - 1)).all()

    all_classes = set(range(dataset.num_classes))

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))
    logits[:, selected_classes] += 1

    logits_mod_single = attributors.SelectedClassesEvidence(
        selected_classes=selected_classes.tolist()
    )(logits, None)

    assert (logits_mod_single[:, selected_classes] > 0).all()
    assert (
        logits_mod_single[:, list(all_classes.difference(selected_classes))] == 0
    ).all()
