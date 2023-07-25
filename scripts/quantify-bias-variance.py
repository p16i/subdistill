import typing
import os
import click
from datetime import datetime

from pathlib import Path
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
import numpy as np

import torch

from xaikd import datasets, models, utils, distillators, constants
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
) -> typing.Tuple[nn.Module, nn.Module]:
    model = models.get_model(teacher)

    utils.deactivate_requires_grad(model)

    random_model = models._resnet18_cifar(num_classes=num_classes)

    if mode == "homogenous":
        new_module = getattr(random_model, layer)
    # elif mode == "inhomogenous":
    #     new_module = getattr(random_model, layer)[:1]
    elif "homogenous-compr" in mode:
        # homogenous-compr0.15
        compr_rate = float(mode.split("compr")[-1])

        dist_info = distillators.get_distill_infor(
            "cifar100-resnet18-p1", layer, compression_rate=compr_rate
        )

        block = distillators.get_approximator_for_resnet18(
            layer, dist_info.num_output_channels
        )[0]

        new_module = nn.Sequential(
            block,
            nn.Conv2d(
                in_channels=dist_info.num_output_channels,
                out_channels=constants.ARCH_LAYER_DIMENSIONS["resnet18"][layer],
                kernel_size=1,
            ),
        )
        print(new_module)
    else:
        raise ValueError(f"mode={mode} not available")

    original_module = getattr(model, layer)
    setattr(model, layer, new_module)

    return model, original_module


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        student_model,
        teacher_module: nn.Module,
        layer: str,
        dataset: datasets.Cifar100SuperClassesDataset,
        train_loader,
        val_loader,
        lambda_mse: float,
        lambda_xent: float,
    ):
        super().__init__()

        (
            self.feat_extractor,
            self.approximator,
            self.classifier,
        ) = models.resnet.split_resnet_18_at(student_model, layer)

        self.teacher_module = teacher_module

        with torch.no_grad():
            x = torch.randn((10, 3, 32, 32))
            assert torch.allclose(
                self.classifier(self.approximator(self.feat_extractor(x))),
                student_model(x),
            )

        self.dataset = dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.arr_metrics = []
        self.lambda_mse = lambda_mse
        self.lambda_xent = lambda_xent
        print(f"Training with lambda_mse={lambda_mse}; lambda_xent={lambda_xent}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.approximator.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, prefix="train")
        return loss

    def eval_safeguard(self):
        self.feat_extractor.eval()
        self.classifier.eval()
        self.teacher_module.eval()

    def on_train_start(self) -> None:
        super().on_train_start()
        self.eval_safeguard()

    def on_validation_start(self) -> None:
        super().on_validation_start()
        self.eval_safeguard()

    def _compute_loss(self, batch, prefix):
        x, y = batch

        assert not self.feat_extractor.training
        assert not self.classifier.training
        assert not self.teacher_module.training

        if prefix == "train":
            assert self.approximator.training

        with torch.no_grad():
            feats = self.feat_extractor(x)

        zh = self.approximator(feats)
        logits = self.classifier(zh)

        loss_mse = self._compute_loss_mse(feats, zh)
        loss_xent = self._compute_loss_xent(logits, y)

        loss = loss_mse + loss_xent

        self.log(f"{prefix}_loss_xent", loss_xent, on_epoch=True)
        self.log(f"{prefix}_loss_mse", loss_mse, on_epoch=True)
        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        return loss

    def _compute_loss_xent(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits[:, self.dataset.selected_classes]
        ynew = self.dataset.transform_target(y)

        return self.lambda_xent * F.cross_entropy(logits, ynew)

    def _compute_loss_mse(
        self, feat_in: torch.Tensor, feat_out: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            expected_feat_out = self.teacher_module(feat_in)

        _, _, h, w = expected_feat_out.shape

        loss = F.mse_loss(feat_out, expected_feat_out, reduction="none")

        loss = loss.flatten(start_dim=1) / (h * w)
        loss = loss.sum(dim=1)

        return self.lambda_mse * loss.mean()

    def validation_step(self, batch, batch_idx):
        self._compute_loss(batch, prefix="val")

    def on_train_epoch_end(self) -> None:
        self.approximator.eval()

        row = dict()

        for name, loader in zip(["train", "val"], [self.train_loader, self.val_loader]):
            acc = metrics.accuracy_with_subclasses(
                nn.Sequential(
                    self.feat_extractor,
                    self.approximator,
                    self.classifier,
                ),
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

        self.approximator.train()


@click.command()
@click.option("--output-dir", type=str, default="./tmp")
@click.option("--seed", default=1)
@click.option("--epochs", default=100)
@click.option("--dataset-name", default="cifar100-people")
@click.option("--mode", default="homogenous")
@click.option("--num-samples", default="50,250,500")
@click.option("--teacher", default="cifar100-resnet18-p1")
@click.option("--lambda-mse", default=1.0)
@click.option("--lambda-xent", default=1.0)
def main(
    teacher,
    dataset_name,
    epochs,
    output_dir,
    seed,
    mode,
    num_samples,
    lambda_mse,
    lambda_xent,
):
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
                output_dir
                / f"{teacher}--layer-{layer}--mode-{mode}--n{num_samples}-lmse{lambda_mse}-lxent{lambda_xent}"
            )

            os.makedirs(log_dir, exist_ok=True)

            student_model, teacher_module = construction_model(
                teacher,
                layer=layer,
                mode=mode,
                num_classes=num_classes,
            )

            model_wrapper = ModelWrapper(
                student_model=student_model,
                teacher_module=teacher_module,
                layer=layer,
                dataset=dataset,
                train_loader=train_loader,
                val_loader=val_loader,
                lambda_mse=lambda_mse,
                lambda_xent=lambda_xent,
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
