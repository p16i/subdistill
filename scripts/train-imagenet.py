import os
import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import argparse

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
import torchvision.models as models

from xaikd import models, datasets, utils

from tqdm import tqdm
from pytorch_lightning.loggers.wandb import WandbLogger

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "xaikd")
WANDB_DIR = os.getenv("WANDB_DIR", "/tmp")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "test")


class ImageNetClassifier(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained model
        self.model = model

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == targets).float().mean()

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", accuracy, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        return [optimizer], [scheduler]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="student-resnet18-d64-128-256-512",
        help="Model architecture",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=90, help="Maximum number of epochs"
    )
    # parser.add_argument(
    #     "--num_workers", type=int, default=16, help="Number of data loading workers"
    # )

    args = parser.parse_args()
    # model = models.get_untrained_model(args.model_name, num_classes=1000)
    model = resnet18(weights=None, num_classes=1000)

    # Initialize model
    model = ImageNetClassifier(model=model)

    dataset = datasets.construct("imagenet")
    ds_train = ImageNet(
        root="/datasets/imagenet",
        split="train",
        transform=dataset.input_training_transformation,
    )
    ds_val = ImageNet(
        root="/datasets/imagenet",
        split="val",
        transform=dataset.input_transformation,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=False,
        drop_last=True,
        # prefetch_factor=4,
    )

    dl_test = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=False,
        drop_last=True,
        # prefetch_factor=4,
    )

    wandb_logger = WandbLogger(
        entity=WANDB_ENTITY,
        save_dir=WANDB_DIR,
        project=WANDB_PROJECT,
        group="2025-09-0.8.x-4.1-reproducing-training-imagenet",
        name=args.model_name,
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=utils.get_device(),
        logger=wandb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    # Train the model
    trainer.fit(model, dl_train, dl_test)


if __name__ == "__main__":
    main()
