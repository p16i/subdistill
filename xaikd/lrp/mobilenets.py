import torch

from torch import nn

from functools import partial

from torchvision.ops import SqueezeExcitation

from torchvision.models.mobilenetv3 import InvertedResidual

from zennit.canonizers import AttributeCanonizer, SequentialMergeBatchNorm
from zennit.composites import Composite

from zennit.rules import Pass
from xaikd.lrp import rules
from xaikd.lrp.nfnets import Summation


def module_map(ctx, name, module, gamma, eps, lb, hb):
    try:
        next(module.children())
    except StopIteration:
        # StopIteration is raised if the iterator has no more elements,
        pass
    else:
        if isinstance(module, SqueezeExcitation):
            return Pass()
        return None

    # count the number of the leaves processed yet in 'leafnum'
    if "leafnum" not in ctx:
        ctx["leafnum"] = 0
    else:
        ctx["leafnum"] += 1

    leafnum = ctx["leafnum"]

    if leafnum == 0:
        return rules.SafeZBox(low=lb.reshape(1, -1, 1, 1), high=hb.reshape(1, -1, 1, 1))
    elif isinstance(module, nn.Hardswish):
        return Pass()
    elif isinstance(module, nn.Linear):
        return rules.SafeGamma(gamma=0.0, stabilizer=eps)
    elif isinstance(module, nn.Conv2d):
        return rules.SafeGamma(gamma=gamma, stabilizer=eps)
    elif isinstance(module, Summation):
        return rules.SafeGammaForPooling(gamma=gamma, stabilizer=eps)
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        return rules.SafeGammaForPooling(gamma=gamma, stabilizer=eps)
    else:
        return None


class InvertedResidualCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, InvertedResidual):
            attributes = {
                "forward": cls.forward.__get__(module),
                "summation": Summation(),
            }

            return attributes

    @staticmethod
    def forward(self, input):
        output = self.block(input)

        if self.use_res_connect:
            concat = torch.stack([output, input], dim=-1)

            output = self.summation(concat)

        return output


class SqueezeExcitationCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, SqueezeExcitation):
            attributes = {
                "forward": cls.forward.__get__(module),
            }

            return attributes

    @staticmethod
    def forward(self, input):
        return self._scale(input.detach()) * input


class EpsilonGammaBox(Composite):
    def __init__(
        self,
        lb: torch.Tensor,
        hb: torch.Tensor,
        gamma=0.1,
        eps=1e-6,
    ):
        super().__init__(
            module_map=partial(
                module_map,
                gamma=gamma,
                eps=eps,
                lb=lb,
                hb=hb,
            ),
            canonizers=[
                SequentialMergeBatchNorm(),
                SqueezeExcitationCanonizer(),
                InvertedResidualCanonizer(),
            ],
        )


def _build_composite(lb, hb, gamma=0.1, eps=1e-12):
    return EpsilonGammaBox(lb=lb, hb=hb, gamma=gamma, eps=eps)
