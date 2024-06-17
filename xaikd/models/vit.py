import typing

import torch
from torch import nn
from torchvision.models import VisionTransformer, vit_b_16, ViT_B_16_Weights

from collections import OrderedDict

from . import register_model


class Lambda(nn.Module):
    def __init__(self, func: typing.Callable) -> None:
        super().__init__()
        self.func = func

    def forward(self, x: torch.Tensor):
        return self.func(x)


class ViTFirstPart(nn.Module):
    def __init__(self, model, layer_ix: int):
        super().__init__()
        assert isinstance(model, VisionTransformer)
        self.model = model
        self.split_encoder = nn.Sequential(model.encoder.layers[: layer_ix + 1])
        self.class_token = model.class_token
        self.heads = model.heads

    def forward(self, x):
        # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L290C2-L299C1
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.split_encoder(
            # https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L156
            x
            + self.model.encoder.pos_embedding
        )

        return x


class ViTSecondPart(nn.Module):
    def __init__(self, model, layer_ix: int):
        super().__init__()
        assert isinstance(model, VisionTransformer)

        self.model = model
        self.split_encoder = nn.Sequential(model.encoder.layers[layer_ix + 1 :])
        self.heads = model.heads

    def forward(self, x):

        # ref: https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py#L298-L303
        x = self.model.encoder.ln(self.split_encoder(x))

        x = x[:, 0]
        x = self.heads(x)
        return x


def split_model_at(
    model: VisionTransformer,
    layer: str,
) -> typing.Tuple[nn.Module, nn.Module]:

    assert isinstance(model, VisionTransformer)

    _, _, layer_ix = layer.split(".")
    layer_ix = int(layer_ix)

    first_part = ViTFirstPart(model, layer_ix)
    second_part = ViTSecondPart(model, layer_ix)

    return first_part, second_part


@register_model("imagenet-vitb-tv")
def _imagenet_vitb() -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    setattr(model, "__last_layer", model.heads.head)
    model.num_classes = 1000

    num_layers = len(model.encoder.layers)

    # native shape:   [bs, #tokens, d]
    # cnn-like shape:  [bs, d, #tokens, 1]

    # convert from cnn-like to native shapes
    transform_to_native_shape = Lambda(lambda x: x.squeeze(3).permute(0, 2, 1))

    # convert from native to cnn-like shape
    transform_to_cnnlike_shape = Lambda(lambda x: x.permute(0, 2, 1).unsqueeze(3))

    for lix in range(num_layers):

        layer = model.encoder.layers[lix]

        if lix == 0:
            steps = [layer, transform_to_cnnlike_shape]
        else:
            steps = [transform_to_native_shape, layer, transform_to_cnnlike_shape]

        model.encoder.layers[lix] = nn.Sequential(*steps)

    model.encoder.ln = nn.Sequential(
        OrderedDict(
            [
                ("_transform_to_native", transform_to_native_shape),
                ("ln", model.encoder.ln),
            ]
        )
    )

    return model
