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
        ds = self.create_dataset(train_split=train_split)

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

    output_dims = models.get_layer_dimensions(model, layer)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=model,
        dataset=dataset,
        layer=layer,
        device=DEVICE,
        number_of_selected_spatial_locations=NUMBER_OF_SPATIAL_LOCATIONS,
        logit_modifier=attributors.OneClassEvidence(dataset),
    )

    assert arr_act.shape == (
        NUMBER_OF_SPATIAL_LOCATIONS * NUMBER_OF_SMALL_DATASET,
        output_dims,
    )

    assert arr_act.shape == arr_ctx.shape


def test_logit_modifier_oneclass():
    dataset = datasets.construct("cifar10")

    all_classes = set(range(dataset.num_classes))
    class1 = 1

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))

    logits_mod_single = attributors.OneClassEvidence(dataset)(
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

    logits_mod_logood = attributors.LogOddEvidence([class1, class2], dataset)(
        logits, target
    )

    assert (logits_mod_logood[:, class1] == logits[:, class1]).all()
    assert (logits_mod_logood[:, class2] == -logits[:, class2]).all()
    assert (
        logits_mod_logood[:, list(all_classes.difference([class1, class2]))] == 0
    ).all()
