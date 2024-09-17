import torch

from torch import nn
from torchvision.models import vision_transformer

from functools import partial

from zennit.rules import Pass, Epsilon
from zennit.canonizers import AttributeCanonizer
from zennit.composites import Composite
from xaikd import nfnetlrp
from copy import deepcopy

from torch.nn.modules.linear import NonDynamicallyQuantizableLinear


class SummationPositionEmbed(torch.nn.Module):
    def __init__(self, pos_embedding):
        super().__init__()
        self.pos_embedding = pos_embedding

    def forward(self, x):
        return x + self.pos_embedding


class EncoderCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, vision_transformer.Encoder):
            attributes = {
                "forward": cls.forward.__get__(module),
                "ln": module.ln,
                "layers": module.layers,
                "sum_pos_embedding": SummationPositionEmbed(module.pos_embedding),
                "dropout": module.dropout,
            }
            return attributes

    @staticmethod
    def forward(self, input):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = self.sum_pos_embedding(input)

        return self.ln(self.layers(self.dropout(input)))


class LayerNormStandardizeStep(nn.Module):
    def __init__(self, eps):

        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        std = (var + self.eps) ** 0.5
        std = std.detach()

        y = (x - mean) / std

        return y


class LayerNormAffineTransformationStep(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return x * self.weight + self.bias


class LayerNormCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, nn.LayerNorm):
            attributes = {
                "forward": cls.forward.__get__(module),
                "adjust": LayerNormAffineTransformationStep(module.weight, module.bias),
                "standardize": LayerNormStandardizeStep(module.eps),
            }
            return attributes

        return None

    @staticmethod
    def forward(self, x):
        x = self.standardize(x)

        x = self.adjust(x)

        return x


class Summation(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=-1)


class MultiHeadAttentionWithoutTransformation(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()

        attn_copied = deepcopy(attn)
        embed_dim = attn_copied.embed_dim

        self.embed_dim = embed_dim

        attn_copied.in_proj_weight = nn.Parameter(torch.cat([torch.eye(embed_dim)] * 3))
        attn_copied.in_proj_bias = nn.Parameter(torch.zeros_like(attn.in_proj_bias))

        attn_copied.out_proj.weight = nn.Parameter(torch.eye(embed_dim))
        attn_copied.out_proj.bias = nn.Parameter(torch.zeros_like(attn.out_proj.bias))

        self.attn = attn_copied

    def forward(self, x):
        embed_dim = self.embed_dim
        query = x[:, :, :embed_dim]
        key = x[:, :, embed_dim : 2 * embed_dim]
        value = x[:, :, 2 * embed_dim :]

        out, _ = self.attn(
            query=query.detach(), key=key.detach(), value=value, need_weights=False
        )
        return out


class AttentionInputProjection(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        return x @ self.weight + self.bias


class EncodingLayerCanonizer(AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, vision_transformer.EncoderBlock):

            embed_dim = module.self_attention.embed_dim

            in_proj = AttentionInputProjection(
                weight=nn.Parameter(module.self_attention.in_proj_weight.T),
                bias=nn.Parameter(module.self_attention.in_proj_bias),
            )

            attributes = {
                "forward": cls.forward.__get__(module),
                "shortcut_summation_1": Summation(),
                "shortcut_summation_2": Summation(),
                "attnwrapper": MultiHeadAttentionWithoutTransformation(
                    module.self_attention
                ),
                "attn_in_proj": in_proj,
                "attn_out_proj": module.self_attention.out_proj,
                "embed_dim": embed_dim,
            }
            return attributes

        return None

    @staticmethod
    def forward(self, input):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )

        x = self.ln_1(input)

        transformed_x = self.attn_in_proj(x)

        out_raw = self.attnwrapper(transformed_x)

        out = self.attn_out_proj(out_raw)

        x = self.dropout(out)

        x = self.shortcut_summation_1(torch.stack([x, input], dim=-1))

        y = self.ln_2(x)
        y = self.mlp(y)

        out = self.shortcut_summation_2(torch.stack([x, y], dim=-1))

        return out


def module_map(ctx, name, module, gamma, eps, lb, hb, first_layer_rule):
    try:
        next(module.children())
    except StopIteration:
        # StopIteration is raised if the iterator has no more elements,
        pass
    else:
        if isinstance(module, MultiHeadAttentionWithoutTransformation):
            return Epsilon(epsilon=0)
        return None

    # count the number of the leaves processed yet in 'leafnum'
    if "leafnum" not in ctx:
        ctx["leafnum"] = 0
    else:
        ctx["leafnum"] += 1

    leafnum = ctx["leafnum"]

    if leafnum == 0:
        if first_layer_rule == "box":
            return nfnetlrp.SafeZBox(
                low=lb.reshape(1, -1, 1, 1), high=hb.reshape(1, -1, 1, 1)
            )
        elif first_layer_rule == "gamma":
            return nfnetlrp.SafeGamma(gamma=gamma, stabilizer=eps)
        else:
            raise
    elif isinstance(module, nn.GELU):
        return Pass()
    elif isinstance(module, SummationPositionEmbed):
        return Pass()
    elif isinstance(module, nn.Linear):
        return nfnetlrp.SafeGamma(gamma=gamma, stabilizer=eps)
    elif isinstance(module, Summation):
        return nfnetlrp.SafeGammaForPooling(gamma=gamma, stabilizer=eps)
    elif isinstance(
        module, (AttentionInputProjection, NonDynamicallyQuantizableLinear)
    ):
        return nfnetlrp.SafeGamma(gamma=gamma, stabilizer=eps)
    elif isinstance(
        module, (LayerNormStandardizeStep, LayerNormAffineTransformationStep)
    ):
        return Epsilon(epsilon=0)
    else:
        return None


class EpsilonGammaBox(Composite):
    def __init__(
        self,
        lb: torch.Tensor,
        hb: torch.Tensor,
        gamma=0.1,
        eps=1e-6,
        first_layer_rule="box",
    ):
        super().__init__(
            module_map=partial(
                module_map,
                gamma=gamma,
                eps=eps,
                lb=lb,
                hb=hb,
                first_layer_rule=first_layer_rule,
            ),
            canonizers=[
                LayerNormCanonizer(),
                EncodingLayerCanonizer(),
                EncoderCanonizer(),
            ],
        )


def _build_composite(
    lb: torch.Tensor,
    hb: torch.Tensor,
    gamma=0.1,
    eps=1e-6,
    first_layer_rule="box",
):
    pass
    return EpsilonGammaBox(
        lb=lb, hb=hb, gamma=gamma, eps=eps, first_layer_rule=first_layer_rule
    )
