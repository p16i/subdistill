import typing
import numpy as np
import pandas as pd
from torch import nn

from torch.utils.data import DataLoader

from xaikd import attributors, metrics, interceptor
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
    assert isinstance(logit_mod, BinaryLogOddWinning)

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
        arr_logodd=arr_logodd,
        logodd_threshold=logit_mod.threshold,
        # remark: this is only used for `Random`
        seed=seed,
    )
    return basis


def evaluate_basis_at_k(
    teacher_model: nn.Module,
    basis: OrthogonalBasis,
    layer: str,
    metric_func: metrics.MetricFunction,
    train_loader: typing.Optional[DataLoader],
    val_loader: typing.Optional[DataLoader],
    arr_ks: typing.List[int],
    device: str,
) -> pd.DataFrame:
    arr_row = []

    for k in arr_ks:
        row = {
            "layer": layer,
            "k": k,
        }
        for data_label, loader in [
            ("train", train_loader),
            ("val", val_loader),
        ]:
            if loader is None:
                continue

            forward_hook = basis.construct_fh_rank_k_projection(k=k, device=device)

            stats = interceptor.attach_projection_forward_hook_at_layer_and_evaluate_metrics(
                model=teacher_model,
                layer=layer,
                dataloader=loader,
                forward_hook_func=forward_hook,
                metric=metric_func,
                device=device,
            )

            for name, value in zip(metric_func._metric_names(), stats):
                row[f"{data_label}_{name}"] = value

        arr_row.append(row)

    df = pd.DataFrame(arr_row)

    return df
