import typing

import numpy as np
import click
from datetime import datetime
import torch
from torch import nn
from torch.nn import functional as F


import wandb

import torchvision
from torchvision.datasets import CelebA
from torchmetrics.classification import BinaryAUROC

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from xaikd import datasets, utils


WANDB_PROJECT = "xaikd-training-teacher-models"
WANDB_GROUP = "celeba"


class ModelWrapper(pl.LightningModule):
    def __init__(self, encoder, lr=1e-3):
        super().__init__()

        self.encoder = encoder

        self.lr = lr

        self.arr_metrics = dict(train=[], val=[])
        for tix in range(datasets.celeba.NUM_CELEBA_ATTRIBUTES):
            self.arr_metrics["train"].append(BinaryAUROC(thresholds=20))
            self.arr_metrics["val"].append(BinaryAUROC(thresholds=20))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    def compute_metric(self, batch, prefix):

        x, y = batch

        arr_logits = self.encoder(x)

        loss = torch.tensor(0.0).to(x.device)
        for tidx in range(datasets.celeba.NUM_CELEBA_ATTRIBUTES):
            y_task = y[:, tidx]
            logits_task = arr_logits[:, tidx]
            attr_loss = F.binary_cross_entropy_with_logits(logits_task, y_task.float())
            loss = loss + attr_loss
            self.arr_metrics[prefix][tidx].update(
                logits_task.detach().cpu(), y_task.detach().long().cpu()
            )

        return loss

    def training_step(self, train_batch, batch_idx):

        loss = self.compute_metric(train_batch, "train")

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        loss = self.compute_metric(batch, "val")

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def summary_metrics(self, prefix):
        arr_values = []
        for tix in range(datasets.celeba.NUM_CELEBA_ATTRIBUTES):
            metric = self.arr_metrics[prefix][tix]
            value = float(metric.compute())

            # value = 1-value if value < 0.5 else value
            arr_values.append(value)
            self.log(f"{prefix}_auroc_attr_{tix}", value)
            metric.reset()

        self.log(f"{prefix}_avg_auroc", float(np.mean(arr_values)))

    def on_validation_epoch_end(self):
        self.summary_metrics("val")

    def on_train_epoch_end(self):
        self.summary_metrics("train")


def model_generator(arch: str) -> typing.Optional[nn.Module]:
    num_outputs = datasets.celeba.NUM_CELEBA_ATTRIBUTES
    model = None
    if arch == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    elif arch == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    elif arch == "vitb16":
        model = torchvision.models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.hidden_dim, num_outputs)
    elif arch == "wideresnet50-2":
        model = torchvision.models.wide_resnet50_2(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_outputs)
    else:
        raise ValueError(f"{arch} doesn't exist")

    return model


@click.command()
@click.option("--arch", type=str)
@click.option("--data-dir", type=str, default="/datasets")
@click.option("--epochs", type=int, default=100)
@click.option("--batch-size", type=int, default=64)
@click.option("--num-workers", type=int, default=12)
def main(arch, data_dir, epochs, batch_size, num_workers):
    arguments = locals()
    start_time = datetime.now()

    dataset = datasets.construct("celeba")

    ds_train = dataset.create_subset(train_split=True)
    ds_train.transform = dataset.input_training_transformation

    ds_val = dataset.create_subset(train_split=False)

    dl_train = datasets.build_dataloader(
        ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dl_val = datasets.build_dataloader(
        ds_val, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    model = model_generator(arch)
    assert model is not None

    wandb_logger = WandbLogger(
        save_dir="/tmp",
        project=WANDB_PROJECT,
        log_model=True,
        group=WANDB_GROUP,
        config=arguments,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=utils.get_device(),
        logger=wandb_logger,
        callbacks=[
            ModelCheckpoint(
                save_last=True,
            )
        ],
        fast_dev_run=True,  # fixme
    )

    trainer.fit(
        ModelWrapper(model),
        dl_train,
        dl_val,
    )

    wandb.finish()

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
