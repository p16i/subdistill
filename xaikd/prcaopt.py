import numpy as np
import numpy.typing as npt

from tqdm import tqdm
import torch

from xaikd import datasets, models

from torch.nn import Sequential
from torch.nn import functional as F
from torch.utils.data import Dataset


@torch.no_grad()
def decorrelate(A: torch.Tensor) -> torch.Tensor:
    # U_ = U @ (U.T @ U)^{-1/2}
    # ref: Hyvärinen et al. (2003), Independent Component Analysis, eq 6.37

    S = A.T @ A
    # somehow, for some classes (e.g. basketball at nfnet-f0's stage1),
    # the eigenvalue decomposition fails if using float.
    # Using double makes it more stable but slightly decreases the speed.
    # See: https://gist.github.com/p16i/4a37e10230c016fcde6c0e571c9ae010
    D, E = torch.linalg.eigh(S.double())
    D = D.float()
    E = E.float()

    inv = E @ torch.diag(1 / (torch.pow(D, 0.5))) @ E.T

    return A @ inv


def learn_prca_opt(
    model: torch.nn.Module,
    layer: str,
    ds_train: Dataset,
    _arr_act: npt.NDArray,
    k: int,
    seed=1,
    epochs=5,
    verbose=False,
    device="cpu",
) -> npt.NDArray:
    rng = torch.Generator()
    rng.manual_seed(seed)

    _, d = _arr_act.shape

    _, eigvecs = np.linalg.eigh(_arr_act.T @ _arr_act)

    U = eigvecs[:, ::-1].copy()
    U = U[:, :k]

    U = torch.from_numpy(U).float().to(device)
    U.requires_grad_(True)

    first_module, second_module = models.split_model_at_layer(model, layer)

    dl_train = datasets.build_dataloader(
        ds_train,
        shuffle=True,
    )

    lr = 1e-3
    acc = 0

    pgb = tqdm(range(epochs))
    for epoch in tqdm(pgb):
        for x, y in dl_train:
            U.grad = None

            x = x.to(device)

            with torch.no_grad():
                expected_logits = model(x)
                act = first_module(x)

            recon = F.conv2d(
                F.conv2d(act, U.T.unsqueeze(2).unsqueeze(3)),
                U.unsqueeze(2).unsqueeze(3),
            )

            actual_logits = second_module(recon)

            loss = torch.linalg.norm(actual_logits - expected_logits, ord=2, dim=1)
            loss = loss.mean()

            loss.backward()

            loss = loss.detach().cpu().numpy()

            pgb.set_description(f"PRCAOpt: k={k}; loss={loss:.4f} ")

            U.data = decorrelate(U - lr * U.grad)

    return U.detach().cpu().numpy()
