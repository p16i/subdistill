import typing

from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from collections import OrderedDict

from xaikd import bases


def construct_mlp(name: str) -> nn.Module:
    # we assume that `name` has `mlp[d+]`.
    assert "mlp" in name

    D = int(name.replace("mlp", ""))

    model = nn.Sequential(
        OrderedDict(
            [
                ("lin1", nn.Linear(3, D)),
                ("act1", nn.ReLU()),
                ("lin2", nn.Linear(D, D // 2)),
                ("act2", nn.ReLU()),
                ("lin3", nn.Linear(D // 2, 6)),
            ]
        )
    )

    setattr(model, "__name", name)

    return model


class ModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.model(x)

        loss = F.cross_entropy(logits, y)

        return loss


def train(
    model,
    train_loader,
    val_loader,
    device: str,
    epochs=5,
):
    pl.seed_everything(1)

    start_time = datetime.now()

    trainer = pl.Trainer(accelerator="cpu", max_epochs=epochs)
    trainer.fit(ModelWrapper(model), train_loader, val_loader)

    print("Time Took:", (datetime.now() - start_time) / 60, "mins")

    return model


def attach_projected_fh_with_k(
    basis: bases.Basis, k: int, device: str
) -> typing.Callable:
    U = basis.artifact["eigvecs"]
    mu = basis.mean

    U = U[:, :k].to(device)

    mu = mu.to(device)

    def fh(mod, input, output):
        assert isinstance(output, torch.Tensor)

        return (output - mu) @ (U @ U.T) + mu

    return fh
