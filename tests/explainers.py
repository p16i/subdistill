import pytest

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image

from xaikd import explainers, models, datasets, attributors, utils, logit_modifiers


def _check_heatmap_finite(explainer_name: str, model_name: str):
    device = utils.get_device()

    dataset = datasets.construct("imagenet-butterfly")

    input_transform = transforms.Normalize(*dataset.input_statistics)

    model = models.get_trained_model(model_name)
    model.to(device)

    explainer = explainers.get_explainer(explainer_name, model, input_transform)

    x = dataset.input_transformation(Image.open("./tests/data/castle.jpg")).unsqueeze(0)
    y = torch.tensor([483])

    ds = TensorDataset(x, y)

    dl = DataLoader(ds)

    arr_logits, arr_heatmaps = explainer.explain(
        dl, logit_modifiers.MultiClassTargetLogit(), device=device
    )

    assert np.isfinite(arr_logits).all()
    assert np.isfinite(arr_heatmaps).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    [
        "imagenet-resnet18-tv",
        "imagenet-resnet50-tv",
        "imagenet-vgg16-tv",
        "imagenet-nfnetf0-dm",
        "imagenet-vitb-tv",
        "imagenet-mobilenetl-tv",
    ],
)
@pytest.mark.parametrize("explainer_name", ["random1", "lrp"])
def test_get_default_explainer(explainer_name, model_name):
    _check_heatmap_finite(explainer_name, model_name)


@pytest.mark.slow
@pytest.mark.parametrize(
    "explainer_name",
    [
        "lrp0.0",
        "lrp0.1",
        "mobilenetlrp0.0",
        "mobilenetlrp0.1",
        "random1",
        "random2",
        "random3",
    ],
)
@pytest.mark.parametrize("model_name", ["imagenet-mobilenetl-tv"])
def test_get_explainer_with_gamma(explainer_name, model_name):
    _check_heatmap_finite(explainer_name, model_name)
