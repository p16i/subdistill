import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from zennit.core import (
    BasicHook,
    Stabilizer,
    expand,
    Hook,
)


from zennit.rules import (
    Pass,
    GammaMod,
    NoMod,
    zero_bias,
    ClampMod,
)
from zennit.types import Linear


from zennit.composites import Composite
from zennit.canonizers import AttributeCanonizer
from functools import partial


from timm.models.layers import SelectAdaptivePool2d, SEModule, pad_same
from timm.models.nfnet import (
    GammaAct,
    ScaledStdConv2dSame,
    NormFreeBlock,
    DownsampleAvg,
)


def lrp_rule_ratio(nom, denom, eps) -> torch.Tensor:
    # Remark: for some reason, torch automatically remove the batch axis of context
    # could this be PyTorch's bug?
    if nom.shape[0] == 1 and len(nom.shape) == 4 and len(nom.shape) == 3:
        output = output.unsqueeze(0)

    # this trick combats getting nan from backprop of x/0.
    # see https://github.com/pytorch/pytorch/issues/4132
    nonzero_ix = denom.abs() > eps

    new_output = torch.zeros_like(nom)

    new_output[nonzero_ix] = nom[nonzero_ix] / denom[nonzero_ix]

    return new_output


class ConstantMul(torch.nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = constant

    def forward(self, x):
        return x * self.constant


class Summation(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=-1)


class SafeGamma(BasicHook):
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):
        mod_kwargs = {"zero_params": zero_params}
        mod_kwargs_nobias = {"zero_params": zero_bias(zero_params)}
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input,
            ],
            param_modifiers=[
                GammaMod(gamma, min=0.0, **mod_kwargs),
                GammaMod(gamma, max=0.0, **mod_kwargs_nobias),
                GammaMod(gamma, max=0.0, **mod_kwargs),
                GammaMod(gamma, min=0.0, **mod_kwargs_nobias),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * lrp_rule_ratio(nom=out_grad, denom=denom, eps=stabilizer)
                    for output, denom in (
                        [(outputs[4] > 0.0, sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0.0, sum(outputs[2:4]))] * 2
                    )
                ]
                + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(
                    input * gradient
                    for input, gradient in zip(inputs[:4], gradients[:4])
                )
            ),
        )


class GammaForPooling(BasicHook):
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):

        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0) * (1 + gamma),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0) * (1 + gamma),
                lambda input: input,
            ],
            param_modifiers=[
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    output * lrp_rule_ratio(nom=out_grad, denom=denom, eps=stabilizer)
                    for output, denom in (
                        [(outputs[4] > 0.0, sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0.0, sum(outputs[2:4]))] * 2
                    )
                ]
                + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(
                    input * gradient
                    for input, gradient in zip(inputs[:4], gradients[:4])
                )
            ),
        )


class SafeGammaForPooling(BasicHook):
    def __init__(self, gamma=0.25, stabilizer=1e-6, zero_params=None):
        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0) * (1 + gamma),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0) * (1 + gamma),
                lambda input: input,
            ],
            param_modifiers=[
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
                NoMod(),
            ],
            output_modifiers=[lambda output: output] * 5,
            gradient_mapper=(
                lambda out_grad, outputs: [
                    # output * out_grad / stabilizer_fn(denom)
                    output * lrp_rule_ratio(nom=out_grad, denom=denom, eps=stabilizer)
                    for output, denom in (
                        [(outputs[4] > 0.0, sum(outputs[:2]))] * 2
                        + [(outputs[4] < 0.0, sum(outputs[2:4]))] * 2
                    )
                ]
                + [torch.zeros_like(out_grad)]
            ),
            reducer=(
                lambda inputs, gradients: sum(
                    input * gradient
                    for input, gradient in zip(inputs[:4], gradients[:4])
                )
            ),
        )


class SafeZBox(BasicHook):
    def __init__(self, low, high, stabilizer=1e-6, zero_params=None):
        def sub(positive, *negatives):
            return positive - sum(negatives)

        mod_kwargs = {"zero_params": zero_params}

        super().__init__(
            input_modifiers=[
                lambda input: input,
                lambda input: expand(low, input.shape, cut_batch_dim=True).to(input),
                lambda input: expand(high, input.shape, cut_batch_dim=True).to(input),
            ],
            param_modifiers=[
                NoMod(**mod_kwargs),
                ClampMod(min=0.0, **mod_kwargs),
                ClampMod(max=0.0, **mod_kwargs),
            ],
            output_modifiers=[lambda output: output] * 3,
            gradient_mapper=(
                lambda out_grad, outputs: (
                    lrp_rule_ratio(out_grad, sub(*outputs), eps=stabilizer),
                )
                * 3
            ),
            reducer=(
                lambda inputs, gradients: sub(
                    *(input * gradient for input, gradient in zip(inputs, gradients))
                )
            ),
        )


class PassWithConstantSign(Hook):
    """Unmodified pass-through rule.
    If the rule of a layer shall not be any other, is elementwise and shall not be the gradient, the `Pass` rule simply
    passes upper layer relevance through to the lower layer.
    """

    def backward(self, module, grad_input, grad_output):
        """Pass through the upper gradient, skipping the one for this layer."""

        if isinstance(module.constant, torch.Tensor):
            sign = torch.sign(module.constant.detach())
        else:
            sign = np.sign(module.constant)

        sign = float(sign)

        return tuple([sign * grad for grad in grad_output])


class ScaledStdConv2dSameCanonizer(AttributeCanonizer):
    """Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models."""

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, ScaledStdConv2dSame):

            std, mean = torch.std_mean(
                module.weight, dim=[1, 2, 3], keepdim=True, unbiased=False
            )
            weight = module.scale * (module.weight - mean) / (std + module.eps)

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
            self.gain * weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def remove(self):
        original_weight = getattr(self.module, "___original_weight")

        super().remove()

        setattr(self.module, "weight", original_weight)

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
        return SafeZBox(low=lb.reshape(1, -1, 1, 1), high=hb.reshape(1, -1, 1, 1))
    elif isinstance(module, (ScaledStdConv2dSame, nn.Linear)):
        return SafeGamma(gamma=gamma, stabilizer=eps)
    elif isinstance(module, (Summation, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        return SafeGammaForPooling(gamma=gamma, stabilizer=eps)
    elif isinstance(module, (GammaAct, ConstantMul)):
        return Pass()
    else:
        return None


class EpsilonGammaBox(Composite):
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
