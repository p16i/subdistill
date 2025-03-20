import torch
import typing

from abc import ABC

import numpy as np

from torch import nn
from torch.nn import functional as F

from xaikd.utils.modules import Centering2D
from xaikd import utils
from xaikd.bases import AdapterMode, OrthogonalBasis
from xaikd.utils.dkd import dkd_loss


from scipy.stats import ortho_group

LAYER_POLICY = dict()


def parse_layer_string(txt: str) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """_summary_

    Args:
        txt (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        teacher_layers : typing.List[str]
        student_layers : typing.List[str]
    """
    teacher_layers = []
    student_layers = []

    for layer in txt.split(","):
        slugs = layer.split(":")

        if len(slugs) == 1:
            teacher_layer = student_layer = slugs[0]
        elif len(slugs) == 2:
            teacher_layer, student_layer = slugs
        else:
            raise ValueError(
                "Could not parse `{layer}` into teacher and student layers!"
            )

        teacher_layers.append(teacher_layer)
        student_layers.append(student_layer)

    assert len(teacher_layers) == len(student_layers)

    return teacher_layers, student_layers


class Policy(nn.Module, ABC):
    transformer_teacher_feats: nn.Module
    transformer_student_feats: nn.Module

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def align_spatial_dimensions(
        self, teacher_feats, student_feats
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return teacher_feats, student_feats

    def forward(self, teacher_feats, student_feats) -> torch.Tensor:
        teacher_feats, student_feats = self.align_spatial_dimensions(
            teacher_feats, student_feats
        )

        transformed_teacher_feats = self.transformer_teacher_feats(teacher_feats)
        transformed_student_feats = self.transformer_student_feats(student_feats)

        assert (
            transformed_student_feats.shape == transformed_teacher_feats.shape
        ), f"{transformed_student_feats.shape}; {transformed_teacher_feats.shape}"

        return self.criterion(transformed_teacher_feats, transformed_student_feats)


def register_layer_policy(name):
    """Decorator to register a layer policy"""

    def wrapped(fn):
        """Wrapped function to register a layer policy provider with`name`"""
        LAYER_POLICY[name] = fn

        return fn

    return wrapped


class LastLayerPolicy(Policy):

    def forward(  # type: ignore
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.criterion(
            teacher_logits=teacher_logits, student_logits=student_logits, target=target
        )

    def criterion(  # type: ignore
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError()


def get_layer_policy(name: str, **kwargs) -> Policy:
    return LAYER_POLICY[name](**kwargs)


def get_last_layer_policy(name: str) -> LastLayerPolicy:
    if name == "kd":
        # ref: temperature value from
        # https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L78
        return KLPolicy(temperature=4.0)
    elif name == "binkd":
        return BinaryKLPolicy()
    elif name == "dkd":
        return DKDPolicy(
            # cf. https://github.com/megvii-research/mdistiller/blob/master/configs/imagenet/r34_r18/dkd.yaml#L29
            temperature=1.0,
            # cf. Section 4.1
            alpha=1.0,
            beta=8.0,
        )
    else:
        raise ValueError(f"Last Layer Policy `{name}` doesn't exist!")


class LayerPolicyCollection(nn.ModuleList):
    def __init__(
        self,
        teacher_layers: typing.List[str],
        student_layers: typing.List[str],
        policies: typing.List[Policy],
    ) -> None:
        super().__init__(policies)

        assert len(teacher_layers) == len(student_layers) == len(policies)

        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.policies = policies

    def forward(self, x):
        pass


class KLPolicy(LastLayerPolicy):
    """
    @article{DBLP:journals/corr/HintonVD15,
        author       = {Geoffrey E. Hinton and
                        Oriol Vinyals and
                        Jeffrey Dean},
        title        = {Distilling the Knowledge in a Neural Network},
        journal      = {CoRR},
        volume       = {abs/1503.02531},
        year         = {2015}
    }

    The implementation is based on the following repositories:
        - https://github.com/yoshitomo-matsubara/torchdistill/blob/45ba679d4649512b520eb4ef7f97b757abf841ee/torchdistill/losses/mid_level.py#L100
        - https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/KD.py
    """

    def __init__(self, temperature: float):
        super().__init__()

        self.T = temperature
        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        kl = F.kl_div(
            torch.log_softmax(student_logits / self.T, dim=1),
            torch.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        )

        # We muliply with scaling factor as mentioned in Hinton et al. (2015)
        # (cf. the paragraph before Section 2.1).
        return (self.T**2) * kl


class BinaryKLPolicy(LastLayerPolicy):

    def __init__(self):
        super().__init__()

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        n = target.shape[0]

        assert teacher_logits.shape == student_logits.shape

        teacher_yp_gv_x = torch.sigmoid(teacher_logits)

        # todo: check whether this is the special os KLDiv
        kl = F.binary_cross_entropy_with_logits(student_logits, teacher_yp_gv_x)

        return kl


class DKDPolicy(LastLayerPolicy):
    """
    @inproceedings{DBLP:conf/cvpr/ZhaoCSQL22,
        author       = {Borui Zhao and
                        Quan Cui and
                        Renjie Song and
                        Yiyu Qiu and
                        Jiajun Liang},
        title        = {Decoupled Knowledge Distillation},
        booktitle    = {{CVPR}},
        pages        = {11943--11952},
        publisher    = {{IEEE}},
        year         = {2022}
    }
    """

    def __init__(
        self,
        temperature: float,
        alpha: float,
        beta: float,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def criterion(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        return dkd_loss(
            logits_student=student_logits,
            logits_teacher=teacher_logits,
            target=target,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        )


class LayerPolicy(Policy):
    def align_spatial_dimensions(self, teacher_feats, student_feats):

        _, _, teacher_height, teacher_width = teacher_feats.shape
        _, _, student_height, student_width = student_feats.shape

        if (student_height == teacher_height) and (student_width == teacher_width == 1):
            # for ViT
            pass
        else:
            # for CNN
            assert teacher_height == teacher_width
            assert student_height == student_width

            if student_height != teacher_height:
                # cf. https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/FT.py#L18C7-L24C17
                # we assume that the feture maps are square.
                assert teacher_height == teacher_width
                assert student_height == student_width

                if student_height > teacher_height:
                    student_feats = F.adaptive_avg_pool2d(
                        student_feats, (teacher_height, teacher_height)
                    )
                elif student_height < teacher_height:
                    teacher_feats = F.adaptive_avg_pool2d(
                        teacher_feats, (student_height, student_height)
                    )

        return teacher_feats, student_feats


@register_layer_policy("nothing")
class NothingPolicy(LayerPolicy):
    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def forward(self, teacher_feats, student_feats):
        return 0


@register_layer_policy("fitnet-relu")
class FitNet(LayerPolicy):
    """
    @inproceedings{DBLP:journals/corr/RomeroBKCGB14,
        author       = {Adriana Romero and
                        Nicolas Ballas and
                        Samira Ebrahimi Kahou and
                        Antoine Chassang and
                        Carlo Gatta and
                        Yoshua Bengio},
        title        = {FitNets: Hints for Thin Deep Nets},
        booktitle    = {{ICLR} (Poster)},
        year         = {2015}
    }
    Remark: the implementation is the FitNet version in VID setup (cf. Ahn et al. (2019, VID, First Paragraph of Section 3).
    The difference from the original setup is that, here, we do NOT pretrain the linear transform of the student.
    Said differently, the FitNet implementation here contains only ONE stage.
    """

    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_student_feats = self._build_student_feats_transfomer(
            teacher_dims=teacher_dims, student_dims=student_dims
        ).to(device)

        self.transformer_teacher_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse

    def _build_student_feats_transfomer(
        self, teacher_dims: int, student_dims: int
    ) -> nn.Module:
        return nn.Sequential(
            # cf.
            # 1. https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/models/util.py#L131
            # 2. https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/adaptation.py#L33
            nn.Conv2d(
                out_channels=teacher_dims,
                in_channels=student_dims,
                # cf. https://github.com/yoshitomo-matsubara/torchdistill/blob/ba62248b48cefeea24f1ef774a870f338711a9d9/configs/sample/ilsvrc2012/fitnet/resnet18_from_resnet152.yaml#L119
                kernel_size=1,
            ),
            nn.ReLU(),
        )


@register_layer_policy("fitnet-noact")
class FitNetTwoLayers(FitNet):
    def _build_student_feats_transfomer(
        self, teacher_dims: int, student_dims: int
    ) -> nn.Module:
        return nn.Conv2d(
            out_channels=teacher_dims,
            in_channels=student_dims,
            kernel_size=1,
        )


@register_layer_policy("vid")
class VIDPolicy(LayerPolicy):
    """
    @inproceedings{DBLP:conf/cvpr/AhnHDLD19,
        author       = {Sungsoo Ahn and
                        Shell Xu Hu and
                        Andreas C. Damianou and
                        Neil D. Lawrence and
                        Zhenwen Dai},
        title        = {Variational Information Distillation for Knowledge Transfer},
        booktitle    = {{CVPR}},
        pages        = {9163--9171},
        publisher    = {Computer Vision Foundation / {IEEE}},
        year         = {2019}
    }
    Remark: The code below is based on the implementation in https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/VID.py#L9C1-L54C20
    """

    # See Supplement  (P4330), the line before the last paragraph.
    init_pred_var = 5.0
    eps = 1e-5

    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        # cf. https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/VID.py#L20
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                stride=stride,
            )

        # cf.
        # - https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L223
        # - https://github.com/yoshitomo-matsubara/torchdistill/blob/e615746172f653ebb59a29b396effba5aa766942/configs/sample/ilsvrc2012/vid/resnet18_from_resnet50.yaml#L123
        num_input_channels = student_dims
        num_mid_channel = num_target_channels = teacher_dims

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )

        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(self.init_pred_var - self.eps) - 1.0)
            * torch.ones(num_target_channels)
        )

        self.eps = self.eps

        self.transformer_student_feats = self.regressor
        self.transformer_teacher_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        target = transformed_teacher_feats
        pred_mean = transformed_student_feats

        # cf. the Paragraph after eq. (5)
        pred_var = torch.log(1.0 + torch.exp(self.log_scale)) + self.eps
        pred_var = pred_var.view(1, -1, 1, 1)

        # cf. eq. (6)
        neg_log_prob = 0.5 * (
            (pred_mean - target) ** 2 / pred_var + torch.log(pred_var)
        )

        # cf. https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py#L53
        # cf. https://github.com/yoshitomo-matsubara/torchdistill/blob/74a710e882a85204ca27233477695d08086ca7b1/torchdistill/losses/mid_level.py#L696
        loss = torch.mean(neg_log_prob)

        return loss


@register_layer_policy("spkd")
class SimilarityPreserveKnowledgeDistillation(LayerPolicy):
    # Tung & Mori 2019, Similarity-Preserving Knowledge Distillation

    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, ord=2
    ) -> None:
        super().__init__()

        self.ord = ord

        class SimilarityMatrix(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, w, h = x.shape

                # eq 2
                x = x.reshape((b, c * w * h))
                x = F.normalize(x, dim=-1)

                return x @ x.T

        self.transformer_student_feats = SimilarityMatrix()
        self.transformer_teacher_feats = SimilarityMatrix()

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:

        b, _ = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # eq 4
        score = torch.norm(
            transformed_teacher_feats - transformed_student_feats, p="fro"
        ) / (b**2)
        return score

    def align_spatial_dimensions(self, teacher_feats, student_feats):
        # remark: we don't do any transformation here.
        return teacher_feats, student_feats


@register_layer_policy("vkd")
class VkD(LayerPolicy):
    # cite...

    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        # todo use this device
        device: str,
    ) -> None:
        super().__init__()

        self._transform_student = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_features=student_dims, out_features=teacher_dims, bias=False)
        )

        def transform_student_fn(feat: torch.Tensor):
            b, d, h, w = feat.shape

            # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L94
            # remark: there, the mean of the student  feature is over sequence of tokens
            # which is last dimensions in our case
            # and it is similar to https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L96
            feat = feat.reshape((b, d, h * w))

            feat = feat.mean(-1)

            return self._transform_student(feat)

        def transform_teacher_fn(feat: torch.Tensor):
            b, d, h, w = feat.shape

            # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L96
            feat = feat.reshape((b, d, h * w))
            feat = feat.mean(-1)

            return F.layer_norm(feat, normalized_shape=(d,))

        self.transformer_student_feats = transform_student_fn
        self.transformer_teacher_feats = transform_teacher_fn

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:

        (
            b,
            _,
        ) = transformed_student_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # remark: they use smooth_l1_loss(...) but why?
        # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L100
        loss_mse = (transformed_student_feats - transformed_teacher_feats) ** 2
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (
            b,
        ), f"shape: {loss_mse.shape}, student: {transformed_student_feats.shape}; teacher: {transformed_teacher_feats.shape}"

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("vkd-modified")
class VkDModified(LayerPolicy):
    # ref: https://arxiv.org/abs/2403.06213

    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        # todo use this device
        device: str,
    ) -> None:
        super().__init__()

        self._transform_student = nn.utils.parametrizations.orthogonal(
            nn.Linear(in_features=student_dims, out_features=teacher_dims, bias=False)
        )

        def transform_student_fn(feat: torch.Tensor):
            b, d, h, w = feat.shape

            return utils.convolve_feature_map_with_linear(feat, self._transform_student)

        def transform_teacher_fn(feat: torch.Tensor):
            b, d, h, w = feat.shape

            # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L96
            feat = feat.reshape((b, d, h * w))
            # shape (b, h*w, d)
            feat = feat.permute((0, 2, 1))

            feat = F.layer_norm(feat, normalized_shape=(d,))

            # shape (b, d, h*w)
            feat = feat.permute((0, 2, 1))
            feat = feat.reshape((b, d, h, w))

            return feat

        self.transformer_student_feats = transform_student_fn
        self.transformer_teacher_feats = transform_teacher_fn

    def criterion(
        self,
        transformed_teacher_feats: torch.Tensor,
        transformed_student_feats: torch.Tensor,
    ) -> torch.Tensor:

        (b, d, w, h) = transformed_student_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # remark: they use smooth_l1_loss(...) but why?
        # cf. https://github.com/roymiles/vkd/blob/4b480506d10bad9bfaf27b144f5929ad4007472d/engine.py#L100
        loss_mse = (transformed_student_feats - transformed_teacher_feats) ** 2
        loss_mse = loss_mse.flatten(start_dim=1) / (w * h)

        assert loss_mse.shape == (
            b,
            d * w * h,
        ), f"shape: {loss_mse.shape}, student: {transformed_student_feats.shape}; teacher: {transformed_teacher_feats.shape}"

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("attention-transfer")
class AttentionTransferPolicy(LayerPolicy):
    """
    @inproceedings{DBLP:conf/iclr/ZagoruykoK17,
        author       = {Sergey Zagoruyko and
                        Nikos Komodakis},
        title        = {Paying More Attention to Attention: Improving the Performance of Convolutional
                        Neural Networks via Attention Transfer},
        booktitle    = {{ICLR} (Poster)},
        publisher    = {OpenReview.net},
        year         = {2017}
    }
    """

    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, ord=2
    ) -> None:
        super().__init__()

        self.ord = ord

        class AttentionMappingFsumP2(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, w, h = x.shape

                # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L19C4-L19C61
                # also Section 3.1 in the paper
                return F.normalize(x.pow(ord).mean(1).view(x.size(0), -1), dim=1, p=ord)

        self.transformer_student_feats = AttentionMappingFsumP2()
        self.transformer_teacher_feats = AttentionMappingFsumP2()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, wh = transformed_teacher_feats.shape

        # comparing spatial dimensions
        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L22
        # Remark: the implementation seems to be different from eq (2) in the paper.
        # In particular, the difference is that calling `.mean()` has a factor of 1/(wh),
        # while the original equal (eq.2) has a factor of `1`.
        loss_mse = (
            (transformed_teacher_feats - transformed_student_feats).pow(self.ord).mean()
        )

        return loss_mse


@register_layer_policy("basis-identity")
class OrthogonalBasisIdentityPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-with-bias-and-scale")
class OrthogonalBasisIdentityBiasAndScalePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class AddBiasAndScale(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(d).reshape(1, d, 1, 1))
                self.scaling = nn.Parameter(torch.tensor(1.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.scaling * x + self.bias

        self.transformer_student_feats = AddBiasAndScale(d=k).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-with-bias")
class OrthogonalBasisIdentityBiasPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        class AddBiasAndScale(nn.Module):
            def __init__(self, d: int):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(d).reshape(1, d, 1, 1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.bias

        self.transformer_student_feats = AddBiasAndScale(d=k).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-bn-only-affine")
class OrthogonalBasisIdentityBatchNormOnlyAffinePolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=False, affine=True
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-bn-only-runstats")
class OrthogonalBasisIdentityBatchNormOnlyRunStatsPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=True, affine=False
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-bn")
class OrthogonalBasisIdentityBatchNormPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=True, affine=True
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.get_scale_factors_for_k(k).max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-bn-wo-normalization")
class OrthogonalBasisIdentityPolicy(LayerPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.BatchNorm2d(
            num_features=k, track_running_stats=True, affine=True
        ).to(device)

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, k, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-rotation")
class OrthogonalBasisRotationPolicy(OrthogonalBasisIdentityPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:

        super().__init__(
            teacher_dims=teacher_dims,
            student_dims=student_dims,
            device=device,
            basis=basis,
        )

        k = student_dims

        class StudenTransform(nn.Module):
            def __init__(self):
                super().__init__()

                self.rotation = nn.utils.parametrizations.orthogonal(
                    nn.Linear(
                        in_features=student_dims, out_features=student_dims, bias=False
                    )
                )
                self.scaling = nn.Parameter(torch.tensor(1.0))

            def forward(self, x):
                x = utils.convolve_feature_map_with_linear(x, self.rotation)
                x = x * self.scaling

                return x

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = StudenTransform()


@register_layer_policy("basis-identity-learnable")
class OrthogonalBasisIdentityLearnablePolicy(OrthogonalBasisIdentityPolicy):
    def __init__(
        self, teacher_dims: int, student_dims: int, device: str, basis: OrthogonalBasis
    ) -> None:
        super().__init__(
            teacher_dims=teacher_dims,
            student_dims=student_dims,
            device=device,
            basis=basis,
        )

        k = student_dims
        W = torch.from_numpy(basis.U[:, :k]).float()

        print("make basis-identitity's weight learnable")
        self.transformer_teacher_feats = nn.Conv2d(
            in_channels=teacher_dims, out_channels=k, kernel_size=1, bias=False
        )
        self.transformer_teacher_feats.weight = nn.Parameter(
            W.T.unsqueeze(2).unsqueeze(3)
        )
        self.transformer_student_feats = nn.Identity()


@register_layer_policy("random")
class OrthogonalRandomPolicy(LayerPolicy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__()

        k = student_dims

        self.U = (
            torch.from_numpy(
                ortho_group.rvs(
                    dim=teacher_dims, random_state=np.random.default_rng(1)
                )[:, :k]
            )
            .float()
            .T.to(device)
            .unsqueeze(2)
            .unsqueeze(3)
        )

        def transform_teacher(x):
            return F.conv2d(x, self.U)

        self.transformer_teacher_feats = transform_teacher

        self.transformer_student_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("randombin")
class OrthogonalRandomPolicy(LayerPolicy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
    ) -> None:
        super().__init__()

        k = student_dims

        self.U = (
            (
                (
                    torch.rand(
                        (k, teacher_dims), generator=torch.Generator().manual_seed(1)
                    )
                    >= 0.5
                )
                / k
            )
            .float()
            .to(device)
            .unsqueeze(2)
            .unsqueeze(3)
        )

        def transform_teacher(x):
            return F.conv2d(x, self.U)

        self.transformer_teacher_feats = transform_teacher

        self.transformer_student_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


@register_layer_policy("basis-attention")
class OrthogonalBasisAttentionPolicy(LayerPolicy):
    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        ord=2,
    ) -> None:
        super().__init__()

        self.ord = ord

        k = student_dims

        self.basis = basis

        class AttentionMappingFsumP2(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, w, h = x.shape

                # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L19C4-L19C61
                # also Section 3.1 in the paper
                return F.normalize(x.pow(ord).mean(1).view(x.size(0), -1), dim=1, p=ord)

        self.transformer_student_feats = AttentionMappingFsumP2()
        self.transformer_teacher_feats = nn.Sequential(
            basis.construct_adapter(k=k, mode=AdapterMode.ENCODER, device=device),
            AttentionMappingFsumP2(),
        )

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, wh = transformed_teacher_feats.shape

        # comparing spatial dimensions
        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L22
        # Remark: the implementation seems to be different from eq (2) in the paper.
        # In particular, the difference is that calling `.mean()` has a factor of 1/(wh),
        # while the original equal (eq.2) has a factor of `1`.
        loss_mse = (
            (transformed_teacher_feats - transformed_student_feats).pow(self.ord).mean()
        )

        return loss_mse


@register_layer_policy("basis-attention20dims")
class OrthogonalBasisAttention20DimsPolicy(LayerPolicy):
    k = 20

    def __init__(
        self,
        teacher_dims: int,
        student_dims: int,
        device: str,
        basis: OrthogonalBasis,
        ord=2,
    ) -> None:
        super().__init__()

        self.ord = ord

        self.basis = basis

        class AttentionMappingFsumP2(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, c, w, h = x.shape

                # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L19C4-L19C61
                # also Section 3.1 in the paper
                return F.normalize(x.pow(ord).mean(1).view(x.size(0), -1), dim=1, p=ord)

        self.transformer_student_feats = AttentionMappingFsumP2()
        self.transformer_teacher_feats = nn.Sequential(
            basis.construct_adapter(k=self.k, mode=AdapterMode.ENCODER, device=device),
            AttentionMappingFsumP2(),
        )

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, wh = transformed_teacher_feats.shape

        # comparing spatial dimensions
        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        # cf. https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L22
        # Remark: the implementation seems to be different from eq (2) in the paper.
        # In particular, the difference is that calling `.mean()` has a factor of 1/(wh),
        # while the original equal (eq.2) has a factor of `1`.
        loss_mse = (
            (transformed_teacher_feats - transformed_student_feats).pow(self.ord).mean()
        )

        return loss_mse


@register_layer_policy("basis-attention1dim")
class OrthogonalBasisAttention1DimPolicy(OrthogonalBasisAttention20DimsPolicy):
    k = 1


@register_layer_policy("basis-identity-cosine")
class OrthogonalBasisIdentityCosinePolicy(OrthogonalBasisIdentityPolicy):
    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        transformed_teacher_feats = F.normalize(transformed_teacher_feats, dim=1)
        transformed_student_feats = F.normalize(transformed_student_feats, dim=1)

        inner_prod = (transformed_teacher_feats * transformed_student_feats).sum(
            dim=1
        ) / (w * h)

        inner_prod = inner_prod.flatten(start_dim=1)

        # sum over all spatial dimensions
        inner_prod = inner_prod.sum(dim=1)

        assert inner_prod.shape == (b,)

        # average over all samples
        inner_prod = inner_prod.mean()

        # converting minimization problem.
        return -inner_prod


@register_layer_policy("basis-innerproduct")
class OrthogonalBasisInnerProductPolicy(OrthogonalBasisIdentityPolicy):
    """
    remark: we might to be careful with this policy for architecturs w/o batchnorm.
    This is because the inner product can grow infinitely large.
    """

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        inner_prod = (transformed_teacher_feats * transformed_student_feats).sum(
            dim=1
        ) / (w * h)

        inner_prod = inner_prod.flatten(start_dim=1)

        # sum over all spatial dimensions
        inner_prod = inner_prod.sum(dim=1)

        assert inner_prod.shape == (b,)

        # average over all samples
        inner_prod = inner_prod.mean()

        # converting minimization problem.
        return -inner_prod


@register_layer_policy("basis-innerproductstudentnorm")
class OrthogonalBasisInnerProductStudentNormPolicy(OrthogonalBasisIdentityPolicy):
    """
    remark: we might to be careful with this policy for architecturs w/o batchnorm.
    This is because the inner product can grow infinitely large.
    """

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        transformed_student_feats = F.normalize(transformed_student_feats, dim=1)

        inner_prod = (transformed_teacher_feats * transformed_student_feats).sum(
            dim=1
        ) / (w * h)

        inner_prod = inner_prod.flatten(start_dim=1)

        # sum over all spatial dimensions
        inner_prod = inner_prod.sum(dim=1)

        assert inner_prod.shape == (b,)

        # average over all samples
        inner_prod = inner_prod.mean()

        # converting minimization problem.
        return -inner_prod


@register_layer_policy("basis-l2detach")
class OrthogonalBasisL2DetachPolicy(OrthogonalBasisIdentityPolicy):
    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        student_norm = torch.linalg.norm(
            transformed_student_feats, dim=1, keepdims=True
        ).detach()

        transformed_student_feats = student_norm * (
            transformed_student_feats / student_norm
        )

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse
