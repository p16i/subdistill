import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from torch.utils.data import DataLoader

from xaikd import attributors
from xaikd.logit_modifiers import BinaryLogOddWinning

from .register import get_basis
from .orthogonal import OrthogonalBasis


def learn_basis(
    teacher_model: nn.Module,
    train_loader: DataLoader,
    logit_mod: BinaryLogOddWinning,
    layer: str,
    basis_name: str,
    device: str,
    seed: int,
) -> OrthogonalBasis:

    rng = np.random.default_rng(seed=seed)

    arr_logodd, arr_act, arr_ctx, mean_act = attributors.extract_activation_grad(
        model=teacher_model,
        layer=layer,
        dataloader=train_loader,
        logit_modifier=logit_mod,
        device=device,
        rng=rng,
    )

    print(f"[layer={layer}] fitting basis={basis_name}")
    basis = get_basis(basis_name)
    basis.fit(
        arr_act=arr_act,
        arr_ctx=arr_ctx,
        mean_act=mean_act,
        arr_logodd=None,
        logodd_threshold=0,
        # remark: this is only used for `Random`
        seed=seed,
    )
    return basis
