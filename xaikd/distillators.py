import copy
import os
import typing


from pytorch_lightning.loggers import TensorBoardLogger


from dataclasses import dataclass
import pytorch_lightning as pl
import numpy as np


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchmetrics


from pathlib import Path

import pandas as pd
from tqdm import tqdm

from xaikd import utils, datasets, attributors, bases, models
from xaikd.utils import metrics


@dataclass
class LayerDistillInfo:
    layer_name: str
    num_input_channels: int
    num_output_channels: int
    # output_spatial_dims: typing.Tuple[int, int]


def get_distill_infor(
    arch: str, layer: str, compression_rate: float
) -> LayerDistillInfo:
    assert arch == "cifar100-resnet18-p1" or arch == "imagenet-resnet18-tv"

    info = dict(
        zip(
            ["layer3", "layer4"],
            [
                LayerDistillInfo(
                    layer_name="layer3",
                    num_input_channels=128,
                    num_output_channels=int(256 * compression_rate),
                    #                    output_spatial_dims=(8, 8),
                ),
                LayerDistillInfo(
                    layer_name="layer4",
                    num_input_channels=256,
                    num_output_channels=int(512 * compression_rate),
                    #                    output_spatial_dims=(4, 4),
                ),
            ],
        )
    )

    return info[layer]


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: nn.Module,
        approximator: nn.Module,
        classification_head: nn.Module,
        lr: float,
        dataset: datasets.Cifar100SuperClassesDataset,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        weight_decay: float,
    ):
        super().__init__()

        self.feature_extrator = feature_extractor
        self.approximator = approximator
        self.classification_head = classification_head

        self.lr = lr

        self.arr_metrics = []
        self.dataset = dataset
        self.val_loss = torchmetrics.MeanMetric()

        # todo: find a better way to do this. Perhaps, via Callback?
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.approximator.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward(self, x) -> torch.Tensor:
        self.feature_extrator.eval()
        self.classification_head.eval()

        with torch.no_grad():
            x = self.feature_extrator(x)

        x = self.approximator(x)

        x = self.classification_head(x)

        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self(x)

        loss = self._compute_loss(logits, y)

        self.log("train_loss", loss)

        return loss

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits[:, self.dataset.selected_classes]
        ynew = self.dataset.transform_target(y)

        return F.cross_entropy(logits, ynew)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits = self(x)

        loss = self._compute_loss(logits, y)

        self.val_loss.update(loss)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute())
        self.val_loss.reset()

    def on_train_epoch_end(self) -> None:
        self.approximator.eval()

        accs = []

        for name, loader in zip(
            ["train", "val"], [self._train_dataloader, self._val_dataloader]
        ):
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

            accs.append(acc)

        self.arr_metrics.append(accs)

        self.approximator.train()


def get_approximator_for_resnet18(
    layer: str, output_dimensions: int, num_classes=100
) -> nn.Module:
    model = models._resnet18_cifar(num_classes)
    model.inplanes = getattr(model, layer)[0].conv1.weight.shape[1]

    return model._make_layer(
        torchvision.models.resnet.BasicBlock,
        output_dimensions,
        1,
        stride=2,
        dilate=False,
    )


class Layerwise:
    def __init__(
        self,
        teacher: torch.nn.Module,
        dataset: datasets.Cifar100SuperClassesDataset,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str,
        weight_decay: float,
    ) -> None:
        self.dataset = dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.teacher = teacher

        self.device = device

        self.ref_acc = metrics.accuracy_with_subclasses(
            self.teacher,
            val_dataloader,
            considered_classes=self.dataset.selected_classes,
            transform_target=self.dataset.transform_target,
            device=self.device,
        )

        self.weight_decay = weight_decay

    def distill(
        self,
        student: nn.Module,
        approx_mod: nn.Module,
        distill_info: LayerDistillInfo,
        epochs: int,
        basis: bases.Basis,
        device: str,
        lr: float,
        log_dir=Path,
    ):
        os.makedirs(str(log_dir), exist_ok=True)

        print(f"Distilling layer={distill_info.layer_name} with {epochs} epochs")

        (
            total_teacher_params,
            _,
        ) = utils.count_params_in_model(self.teacher)

        self.on_training_layer_start(
            student, approx_mod=approx_mod, distill_info=distill_info
        )

        count_total_params, count_trainable_params = utils.count_params_in_model(
            student
        )

        print(distill_info)
        print(
            f"> total_params: {count_total_params} (trainable {count_trainable_params})"
        )
        print(
            f"> compression w.r.t. to teacher: {count_total_params/total_teacher_params*100:.2f}% ({count_total_params:.2e}/{total_teacher_params:.2e}) "
        )

        assert (
            count_trainable_params > 0 and count_trainable_params < total_teacher_params
        )

        decoder = basis.contruct_rank_d_decoder(
            distill_info.num_output_channels, device=device
        )

        approx_mod.adapter = decoder

        feature_extractor, _, classification_head = models.resnet.split_resnet_18_at(
            student, distill_info.layer_name
        )

        training_wrapper = ModelWrapper(
            feature_extractor=feature_extractor,
            approximator=approx_mod,
            classification_head=classification_head,
            lr=lr,
            dataset=self.dataset,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            weight_decay=self.weight_decay,
        )

        student.to(device)
        student_acc_before_training = metrics.accuracy_with_subclasses(
            student,
            dl=self.val_dataloader,
            considered_classes=self.dataset.selected_classes,
            transform_target=self.dataset.transform_target,
            device=self.device,
        )

        print(
            f"Student ACC Before Training: {student_acc_before_training:.4f} (teacher={self.ref_acc:.4f})"
        )

        print(f"Training log is saved to `{log_dir}`")

        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=epochs,
            logger=TensorBoardLogger(log_dir),
            log_every_n_steps=1,
            enable_checkpointing=False,
            deterministic=True,
        )
        trainer.fit(training_wrapper, self.train_dataloader, self.val_dataloader)

        arr_metrics = []

        for epoch, (train_acc, val_acc) in enumerate(training_wrapper.arr_metrics):
            arr_metrics.append(
                dict(
                    layer=distill_info.layer_name,
                    epoch=epoch,
                    epoch_val_acc=val_acc,
                    epoch_train_acc=train_acc,
                    teacher_acc=self.ref_acc,
                    student_acc_before_training=student_acc_before_training,
                    student_trainable_param=count_trainable_params,
                    student_total_params=count_total_params,
                    teacher_total_params=total_teacher_params,
                )
            )

        return arr_metrics

    def on_training_layer_start(
        self,
        student: nn.Module,
        approx_mod: nn.Module,
        distill_info: LayerDistillInfo,
    ) -> nn.Module:
        utils.deactivate_requires_grad(student)

        setattr(student, distill_info.layer_name, approx_mod)

        return approx_mod
