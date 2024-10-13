import torch
from torch import nn
from torch.nn import functional as F


from zennit.rules import Pass
from xaikd.lrp import rules

from zennit.composites import Composite
from zennit.canonizers import AttributeCanonizer
from functools import partial


from timm.models.layers import SEModule, pad_same
from timm.models.nfnet import (
    GammaAct,
    ScaledStdConv2dSame,
    NormFreeBlock,
)


class ConstantMul(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x * self.constant


class Summation(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=-1)


class ScaledStdConv2dSameCanonizer(AttributeCanonizer):
    """Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models."""

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module: ScaledStdConv2dSame):
        if isinstance(module, ScaledStdConv2dSame):

            weight = F.batch_norm(
                module.weight.reshape(1, module.out_channels, -1),
                None,
                None,
                weight=(module.gain * module.scale).view(-1),
                training=True,
                momentum=0.0,
                eps=module.eps,
            ).reshape_as(module.weight)

            attributes = {
                "forward": cls.forward.__get__(module),
                "weight": nn.Parameter(weight),
                "___original_weight": module.weight,
            }
            return attributes

        return None

    @staticmethod
    def forward(self, x):
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)

        weight = self.weight
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def remove(self):
        original_weight = getattr(self.module, "___original_weight")

        # remove all the keys
        for key in self.attribute_keys:
            delattr(self.module, key)

        # rollback to the original weight
        setattr(self.module, "weight", original_weight)

    def copy(self):
        # pat: we have to do this, otherwise copy() comes from AttributeCanonizer
        return ScaledStdConv2dSameCanonizer()


class SEModuleCanonizer(AttributeCanonizer):
    """Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models."""

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SEModule):
            attributes = {
                "forward": cls.forward.__get__(module),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        x_se = x.detach().mean((2, 3), keepdim=True)

        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)

        gate = self.gate(x_se)

        out = x * gate

        return out


class NormFreeBlockCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, NormFreeBlock):
            attributes = {
                "forward": cls.forward.__get__(module),
                "shortcut_summation": Summation(),
                "mul_alpha": ConstantMul(module.alpha),
                "mul_beta": ConstantMul(module.beta),
                "mul_attn_gain": ConstantMul(module.attn_gain),
                "mul_skipinit_gain": ConstantMul(module.skipinit_gain),
            }
            return attributes

        return None

    @staticmethod
    def forward(self, x):
        # @todo: add reference to this function
        shortcut = x

        out_act1 = self.mul_beta(self.act1(x))

        residue_inp = out_act1

        # residual branch
        out = self.conv1(residue_inp)
        out = self.conv2(self.act2(out))
        if self.conv2b is not None:
            out = self.conv2b(self.act2b(out))
        if self.attn is not None:
            raise
            out = self.attn_gain * self.attn(out)

        out_conv3 = self.conv3(self.act3(out))

        if self.attn_last is not None:
            out_attn_gain = self.mul_attn_gain(self.attn_last(out_conv3))

        out = self.drop_path(out_attn_gain)

        if self.skipinit_gain is not None:
            # @pat: why does this value matter? what is the inuition the authors of NFNets?
            out = self.mul_skipinit_gain(out)

        residue_output = self.mul_alpha(out)

        if self.downsample is not None:
            inp_shortcut = out_act1
            shortcut = self.downsample(inp_shortcut)

        concat = torch.stack([residue_output, shortcut], dim=-1)

        # this is the only line that is different from the original function.
        out = self.shortcut_summation(concat)

        return out


def module_map(ctx, name, module, gamma, eps, lb, hb):
    try:
        next(module.children())
    except StopIteration:
        # StopIteration is raised if the iterator has no more elements,
        pass
    else:
        # todo: find a way to handle this case outside try
        if isinstance(module, SEModule):
            return Pass()

        # if StopIteration is not raised on the first element, module is not a leaf
        return None

    # count the number of the leaves processed yet in 'leafnum'
    if "leafnum" not in ctx:
        ctx["leafnum"] = 0
    else:
        ctx["leafnum"] += 1

    leafnum = ctx["leafnum"]

    if leafnum == 0:
        assert isinstance(module, ScaledStdConv2dSame)
        return rules.SafeZBox(low=lb.reshape(1, -1, 1, 1), high=hb.reshape(1, -1, 1, 1))
    elif isinstance(module, (ScaledStdConv2dSame, nn.Linear)):
        return rules.SafeGamma(gamma=gamma, stabilizer=eps)
    elif isinstance(module, (Summation, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        return rules.SafeGammaForPooling(gamma=gamma, stabilizer=eps)
    elif isinstance(module, (GammaAct, ConstantMul)):
        return Pass()
    else:
        return None


class NFNetComposite(Composite):
    def __init__(
        self,
        lb: torch.Tensor,
        hb: torch.Tensor,
        gamma=0.1,
        eps=0.01,
    ):
        super().__init__(
            module_map=partial(module_map, gamma=gamma, eps=eps, lb=lb, hb=hb),
            canonizers=[
                NormFreeBlockCanonizer(),
                SEModuleCanonizer(),
                ScaledStdConv2dSameCanonizer(),
            ],
        )


def _build_composite(
    lb: torch.Tensor,
    hb: torch.Tensor,
    gamma=0.1,
    eps=1e-2,
) -> Composite:
    return NFNetComposite(
        lb=lb,
        hb=hb,
        gamma=gamma,
        eps=eps,
    )
