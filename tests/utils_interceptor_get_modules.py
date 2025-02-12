import pytest

from collections import OrderedDict
from torch import nn
from xaikd import models
from xaikd.utils import interceptor


class RefModule(nn.Module):
    pass


REF_MODULE = RefModule()


class FlatWithStrModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.layer1 = nn.Dropout()
        self.layer2 = REF_MODULE


class FlatWithIndexModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.preprocessing = nn.Dropout()
        self.layers = nn.Sequential(nn.Dropout(), REF_MODULE)


class ThreeLevelWithIndexModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = FlatWithIndexModel()


class ThreeLevelWithStrModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = nn.Sequential(OrderedDict([("features", FlatWithStrModel())]))


@pytest.mark.parametrize(
    "model,layer_str",
    [
        (
            FlatWithStrModel(),
            "layer2",
        ),
        (
            FlatWithIndexModel(),
            "layers.1",
        ),
        (
            ThreeLevelWithIndexModel(),
            "encoder.layers.1",
        ),
        (
            ThreeLevelWithStrModel(),
            "encoder.features.layer2",
        ),
    ],
)
def test_get_module(model: nn.Module, layer_str: str):

    actual = interceptor.get_module(model, layer_str=layer_str)

    assert actual == REF_MODULE


def test_get_module_from_real_model__resnet():

    model = models.get_trained_model("imagenet-resnet18-tv")
    layer_str = "layer3"

    actual = interceptor.get_module(model, layer_str=layer_str)
    expected = model.layer3

    assert actual == expected


def test_get_module_from_real_model__vgg16():

    model = models.get_trained_model("imagenet-vgg16-tv")
    layer_str = "features.27"

    actual = interceptor.get_module(model, layer_str=layer_str)
    expected = getattr(model, "features")[27]

    assert actual == expected


def test_get_module_from_real_model__nfnet():

    model = models.get_trained_model("imagenet-nfnetf0-dm")
    layer_str = "stages.0"

    actual = interceptor.get_module(model, layer_str=layer_str)
    expected = getattr(model, "stages")[0]

    assert actual == expected


def test_get_module_from_real_model__vitb():

    model = models.get_trained_model("imagenet-vitb-tv")
    layer_str = "encoder.layers.8"

    actual = interceptor.get_module(model, layer_str=layer_str)
    expected = getattr(getattr(model, "encoder"), "layers")[8]

    assert actual == expected
