import os
import click
from datetime import datetime

from pathlib import Path
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np

import torch

from xaikd import datasets, models, utils, distillators
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


def construction_model(
    teacher: str, layer: str, mode: str, num_classes: int
) -> nn.Module:
    model = models.get_model(teacher)

    utils.deactivate_requires_grad(model)

    random_model = models._resnet18_cifar(num_classes=num_classes)

    if mode == "homogenous":
        new_module = getattr(random_model, layer)
    elif mode == "inhomogenous":
        new_module = getattr(random_model, layer)[:1]
    elif "homogenous-compr" in mode:
        # homogenous-compr0.15
        compr_rate = float(mode.split("compr")[-1])

        dist_info = distillators.get_distill_infor(
            "cifar100-resnet18-p1", layer, compression_rate=compr_rate
        )

        new_module = distillators.get_approximator_for_resnet18(
            layer, dist_info.num_output_channels
        )
    else:
        raise ValueError("mode={mode} not available")

    setattr(model, layer, new_module)

    return model


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        dataset: datasets.Cifar100SuperClassesDataset,
        train_loader,
        val_loader,
    ):
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

        logits = self.model(x)

        loss = self._compute_loss(logits, y)

        self.log("train_loss", loss)

        return loss

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits[:, self.dataset.selected_classes]
        ynew = self.dataset.transform_target(y)

        return F.cross_entropy(logits, ynew)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits = self.model(x)
        loss = self._compute_loss(logits, y)

        self.val_loss.update(loss)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute())
        self.val_loss.reset()

    def on_train_epoch_end(self) -> None:
        self.model.eval()

        row = dict()

        for name, loader in zip(["train", "val"], [self.train_loader, self.val_loader]):
            acc = metrics.accuracy_with_subclasses(
                self,
                loader,
                considered_classes=self.dataset.selected_classes,
                transform_target=self.dataset.transform_target,
                device=self.device,
            )

            self.logger.experiment.add_scalar(
                f"{name}_acc", acc, global_step=self.current_epoch
            )

            row[f"{name}_acc"] = acc

        self.arr_metrics.append(row)

        self.model.train()


@click.command()
@click.option("--output-dir", type=str, default="./tmp")
@click.option("--seed", default=1)
@click.option("--epochs", default=50)
@click.option("--dataset-name", default="cifar100-people")
@click.option("--mode", default="homogenous")
@click.option("--num-samples", default="5,50,250,500")
@click.option("--teacher", default="cifar100-resnet18-p1")
def main(teacher, dataset_name, epochs, output_dir, seed, mode, num_samples):
    arguments = locals()

    layers = ["layer3", "layer4"]

    assert "cifar100" in dataset_name
    num_classes = 100

    start_time = datetime.now()

    output_dir = Path(output_dir) / f"{dataset_name}-seed{seed}"

    device = utils.get_device()

    os.makedirs(output_dir, exist_ok=True)

    arr_num_samples = np.array(num_samples.split(",")).astype(int).tolist()

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

        for layer in layers:
            log_dir = (
                output_dir / f"{teacher}--layer-{layer}--mode-{mode}--n{num_samples}"
            )

            os.makedirs(log_dir, exist_ok=True)

            model = construction_model(
                teacher,
                layer=layer,
                mode=mode,
                num_classes=num_classes,
            )

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
