import typing

from collections import OrderedDict
import timm


from timm.models.nfnet import NormFreeNet
from torch import nn

from . import register_model


def split_model_at(
    model: NormFreeNet, layer: str
) -> typing.Tuple[nn.Sequential, nn.Sequential]:

    assert isinstance(model, NormFreeNet)

    _, stage_ix = layer.split(".")
    stage_ix = int(stage_ix)

    head = nn.Sequential(
        OrderedDict([("stem", model.stem), ("stages", model.stages[: (stage_ix + 1)])])
    )

    tail = nn.Sequential(
        OrderedDict(
            [
                ("stages", model.stages[stage_ix + 1 :]),
                ("final_conv", model.final_conv),
                ("final_act", model.final_act),
                ("head", model.head),
            ]
        )
    )

    return head, tail


def create_nfnet(name):
    teacher = timm.create_model(name, pretrained=True).eval()
    setattr(teacher, "__last_layer", teacher.head.fc)
    return teacher


@register_model("imagenet-nfnetf0-dm")
def _nfnetf0():
    return create_nfnet("dm_nfnet_f0")
