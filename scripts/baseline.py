import os
import click
from datetime import datetime

from pathlib import Path
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np

import torch

from xaikd import datasets, models, utils
from xaikd.utils import metrics
from torch import nn
from torchvision.models import resnet
from torchvision import transforms

from torch.nn import functional as F

import torchmetrics


def get_transformation(dataset_name):
    if "cifar100" in dataset_name:
        return [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    elif "imagenet" in dataset_name:
        return [
            transforms.RandomHorizontalFlip(),
        ]
    else:
        raise NotImplementedError("")


class Lenet5(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 6, kernel_size=5, padding=2, padding_mode="replicate"
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.AvgPool2d(kernel_size=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lin1 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()

        self.lin2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.act4(self.lin1(x))

        x = self.lin2(x)

        return x


def construction_model(name: str, dataset_name: str, num_classes: int) -> nn.Module:
    if name == "lenet5":
        return Lenet5(num_classes=num_classes)
    elif "resnet" in name:
        model = None

        if "pretrain" in name:
            if "cifar100" in dataset_name:
                model = models.get_model("cifar100-resnet18-p1")
            elif "imagenet" in dataset_name:
                model = models.get_model("imagenet-resnet18-tv")
            utils.deactivate_requires_grad(model)
        else:
            if "cifar100" in dataset_name:
                model = models._resnet18_cifar(num_classes=num_classes)
            elif "imagenet" in dataset_name:
                model = resnet.resnet18(weights=None)

        if "resnet18-2l" in name:
            model.layer3 = nn.Identity()
            model.layer4 = nn.Identity()
            model.fc = nn.Linear(128, num_classes)
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif "resnet18-3l" in name:
            model.layer4 = nn.Identity()
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            model.fc = nn.Linear(256, num_classes)
        elif name == "resnet18-full":
            pass
        else:
            raise ValueError(f"model={name} not available")

        return model

    else:
        raise ValueError(f"model={name} not available")


class ModelWrapper(pl.LightningModule):
    def __init__(self, model, dataset, train_loader, val_loader):
        super().__init__()

        self.model = model

        self.val_loss = torchmetrics.MeanMetric()
        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.arr_metrics = []

    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        yh = self.model(x)

        loss = F.cross_entropy(yh, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits = self.model(x)

        loss = F.cross_entropy(logits, y)
        self.val_loss.update(loss)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute())
        self.val_loss.reset()

    def on_train_epoch_end(self) -> None:
        self.model.eval()

        row = dict()

        for name, loader in zip(["train", "val"], [self.train_loader, self.val_loader]):
            auroc, _ = metrics.auroc(
                self,
                loader,
                self.dataset.selected_classes,
                self.device,
                should_convert_auroc=True,
            )

            self.logger.experiment.add_scalar(
                f"{name}_auroc", auroc, global_step=self.current_epoch
            )

            row[f"{name}_auroc"] = auroc

        self.arr_metrics.append(row)

        self.model.train()


@click.command()
@click.option("--output-dir", type=str, default="./tmp")
@click.option("--seed", default=1)
@click.option("--epochs", default=100)
@click.option("--dataset-name", default="cifar100-35vs98")
@click.option("--num-samples", default="10,100,1000")
def main(dataset_name, epochs, output_dir, seed, num_samples):
    arguments = locals()
    start_time = datetime.now()

    output_dir = Path(output_dir) / f"{dataset_name}-seed{seed}"

    device = utils.get_device()

    os.makedirs(output_dir, exist_ok=True)

    arr_num_samples = np.array(num_samples.split(",")).astype(int).tolist()

    num_classes = 100 if "cifar100" in dataset_name else 1000

    for num_samples in arr_num_samples:
        pl.seed_everything(seed)
        dataset = datasets.construct(dataset_name, num_training_samples=num_samples)
        train_loader = dataset.loader(train_split=True, shuffle=True)
        val_loader = dataset.loader(train_split=False)

        train_loader.dataset.dataset.transform = transforms.Compose(
            [
                *get_transformation(dataset_name=getattr(dataset, "__name")),
                train_loader.dataset.dataset.transform,
            ]
        )

        for model_name in [
            "lenet5",
            "resnet18-2l",
            "pretrained-resnet18-2l",
            "resnet18-3l",
            "pretrained-resnet18-3l",
            # "resnet18-full",
        ]:
            log_dir = output_dir / f"{model_name}-n{num_samples}"
            os.makedirs(log_dir, exist_ok=True)

            click.echo(f"Working on `{model_name}`")

            model = construction_model(
                model_name,
                dataset_name=dataset_name,
                num_classes=num_classes,
            )

            assert model is not None

            model_wrapper = ModelWrapper(
                model=model,
                dataset=dataset,
                train_loader=train_loader,
                val_loader=val_loader,
            )

            trainer = pl.Trainer(
                accelerator=device,
                max_epochs=epochs,
                logger=TensorBoardLogger(log_dir),
                log_every_n_steps=1,
                enable_checkpointing=False,
                deterministic=True,
            )
            trainer.fit(model_wrapper, train_loader, val_loader)

            df_stats = pd.DataFrame(model_wrapper.arr_metrics)

            df_stats.to_csv(log_dir / "stats.csv")

    click.echo(f"Check results at {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
