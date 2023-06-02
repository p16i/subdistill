import typing

from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

import torchmetrics


from collections import OrderedDict

from xaikd import bases
from xaikd.toy import data


def construct_mlp(name: str) -> nn.Module:
    # we assume that `name` has `mlp[d+]`.
    assert "mlp" in name

    D = int(name.replace("mlp", ""))

    model = nn.Sequential(
        OrderedDict(
            [
                ("lin1", nn.Linear(2, D)),
                ("act1", nn.ReLU()),
                ("lin2", nn.Linear(D, D // 2)),
                ("act2", nn.ReLU()),
                ("lin3", nn.Linear(D // 2, data.NUM_CLASSES)),
            ]
        )
    )

    setattr(model, "__name", name)

    return model


class ModelWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model

        num_classes = self.model[-1].bias.shape[-1]

        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.model(x)

        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits = self.model(x)

        self.valid_acc.update(logits, y)

    def on_validation_epoch_end(self):
        acc = self.valid_acc.compute()
        self.log("valid_acc_epoch", self.valid_acc.compute())
        self.valid_acc.reset()


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


class QDA(nn.Module):
    def __init__(self, centroids, covs, device) -> None:
        super().__init__()

        self.centroids = torch.tensor(centroids).to(device).float()
        self.num_classes = self.centroids.shape[0]

        self.covs = torch.tensor(covs).to(device).float()

    def forward(self, x):
        logits = torch.zeros(x.shape[0], self.num_classes).to(x.device)

        for cix in range(self.num_classes):
            # the code is taken from the reference below
            # ref: https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/discriminant_analysis.py#L941
            mu = self.centroids[cix, :]

            xm = x - mu
            V, U = torch.linalg.eigh(self.covs[cix, :, :])

            x2 = xm @ (U * (V ** (-0.5)))

            norm2 = torch.sum(x2**2, dim=1)
            u = torch.log(V).sum()

            logits[:, cix] = -0.5 * (norm2 + u)

        return logits
