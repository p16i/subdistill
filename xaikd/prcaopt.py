import numpy as np
import numpy.typing as npt

from tqdm import tqdm
import torch

from xaikd import datasets, models

from torch.nn import Sequential
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.parametrizations import orthogonal


def learn_prca_opt(
    model: torch.nn.Module,
    layer: str,
    dataloader: DataLoader,
    Uinit: npt.NDArray,
    k: int,
    seed=1,
    epochs=5,
    verbose=False,
    device="cpu",
) -> npt.NDArray:
    rng = torch.Generator()
    rng.manual_seed(seed)

    d, kp = Uinit.shape

    assert kp == k

    linear_layer = torch.nn.Linear(k, d, bias=False)

    linear_layer.weight = torch.nn.Parameter(torch.from_numpy(Uinit.T).float())

    ortho_layer = orthogonal(linear_layer).to(device)

    first_module, second_module = models.split_model_at_layer(model, layer)

    lr = 1e-3

    optimizer = torch.optim.Adam(ortho_layer.parameters(), lr=lr)

    pgb = tqdm(range(epochs))
    for epoch in tqdm(pgb):
        for x, y in dataloader:
            optimizer.zero_grad()

            U = ortho_layer.weight.T

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
            optimizer.step()

            loss = loss.detach().cpu().numpy()

            pgb.set_description(f"PRCAOpt: k={k}; loss={loss:.4f} ")

    return ortho_layer.weight.T.detach().cpu().numpy()
