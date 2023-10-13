import torch
import typing

from abc import ABC

import numpy as np

from torch import nn
from torch.nn import functional as F

from xaikd.utils.modules import Centering2D
from xaikd.bases import Basis, AdapterMode


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

        assert transformed_student_feats.shape == transformed_teacher_feats.shape

        return self.criterion(transformed_teacher_feats, transformed_student_feats)


def register_layer_policy(name):
    """Decorator to register a layer policy"""

    def wrapped(fn):
        """Wrapped function to register a layer policy provider with`name`"""
        LAYER_POLICY[name] = fn

        return fn

    return wrapped


def get_layer_policy(name: str, **kwargs) -> Policy:
    return LAYER_POLICY[name](**kwargs)


class LayerPolicyCollection(nn.ModuleList):
    def __init__(
        self,
        teacher_layers: typing.List[str],
        student_layers: typing.List[str],
        policies: typing.Iterable[Policy],
    ) -> None:
        super().__init__(policies)

        assert len(teacher_layers) == len(student_layers) == len(policies)

        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.policies = policies

    def forward(self, x):
        pass


class KLPolicy(Policy):
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
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor
    ) -> torch.Tensor:
        kl = F.kl_div(
            torch.log_softmax(student_logits / self.T, dim=1),
            torch.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        )

        # We muliply with scaling factor as mentioned in Hinton et al. (2015)
        # (cf. the paragraph before Section 2.1).
        return (self.T**2) * kl


class LayerPolicy(Policy):
    def align_spatial_dimensions(self, teacher_feats, student_feats):
        # we assume that the feture maps are square.
        assert (np.array(teacher_feats.shape[2:]) == teacher_feats.shape[2]).all() and (
            np.array(student_feats.shape[2:]) == student_feats.shape[2]
        ).all()

        _, _, teacher_height, _ = teacher_feats.shape
        _, _, student_height, _ = student_feats.shape

        # cf. https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/distiller_zoo/FT.py#L18C7-L24C17
        if student_height > teacher_height:
            student_feats = F.adaptive_avg_pool2d(
                student_feats, (teacher_height, teacher_height)
            )
        elif student_height < teacher_height:
            teacher_feats = F.adaptive_avg_pool2d(
                teacher_feats, (student_height, student_height)
            )
        else:
            pass

        return teacher_feats, student_feats


@register_layer_policy("nothing")
class NothingPolicy(LayerPolicy):
    def __init__(self, teacher_dims: int, student_dims: int, device: str) -> None:
        super().__init__()

        self.transformer_teacher_feats = nn.Identity()
        self.transformer_student_feats = nn.Identity()

    def forward(self, teacher_feats, student_feats):
        return 0


@register_layer_policy("fitnet")
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
            # cf: https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/adaptation.py#L34
            nn.BatchNorm2d(num_features=teacher_dims),
            # cf. https://github.com/yoshitomo-matsubara/torchdistill/blob/ba62248b48cefeea24f1ef774a870f338711a9d9/configs/sample/ilsvrc2012/fitnet/resnet18_from_resnet152.yaml#L122
            nn.ReLU(),
        )


@register_layer_policy("fitnet-2l")
class FitNetTwoLayers(FitNet):
    def _build_student_feats_transfomer(
        self, teacher_dims: int, student_dims: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                out_channels=teacher_dims,
                in_channels=student_dims,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=teacher_dims),
            nn.ReLU(),
            nn.Conv2d(
                out_channels=teacher_dims,
                in_channels=teacher_dims,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=teacher_dims),
            nn.ReLU(),
        )


@register_layer_policy("fitnet-3l")
class FitNetThreeLayers(FitNet):
    def _build_student_feats_transfomer(
        self, teacher_dims: int, student_dims: int
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                out_channels=teacher_dims,
                in_channels=student_dims,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=teacher_dims),
            nn.ReLU(),
            nn.Conv2d(
                out_channels=teacher_dims,
                in_channels=teacher_dims,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=teacher_dims),
            nn.ReLU(),
            nn.Conv2d(
                out_channels=teacher_dims,
                in_channels=teacher_dims,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_features=teacher_dims),
            nn.ReLU(),
        )


@register_layer_policy("fitnet-cosine")
class FitNetCosineSetup(FitNet):
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
        self, teacher_dims: int, student_dims: int, device: str, basis: Basis
    ) -> None:
        super().__init__()

        k = student_dims

        self.basis = basis
        self.transformer_teacher_feats = basis.construct_adapter(
            k=k, mode=AdapterMode.ENCODER, device=device
        )

        self.transformer_student_feats = nn.Identity()

    def criterion(self, transformed_teacher_feats, transformed_student_feats):
        b, _, w, h = transformed_teacher_feats.shape

        assert transformed_teacher_feats.shape == transformed_student_feats.shape

        loss_mse = F.mse_loss(
            transformed_student_feats, transformed_teacher_feats, reduction="none"
        ) / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse / self.basis.artifact["scale"].max()

        # sum over all spatial dimensions
        loss_mse = loss_mse.sum(dim=1)

        assert loss_mse.shape == (b,)

        # average over all samples
        loss_mse = loss_mse.mean()

        return loss_mse


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
        self, teacher_dims: int, student_dims: int, device: str, basis: Basis, ord=2
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
        self, teacher_dims: int, student_dims: int, device: str, basis: Basis, ord=2
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
