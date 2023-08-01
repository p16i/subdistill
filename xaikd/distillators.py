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

from enum import Enum

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
        teacher_module: nn.Module,
        adapter: nn.Module,
        approximator: nn.Module,
        classification_head: nn.Module,
        lr: float,
        dataset: datasets.Cifar100SuperClassesDataset,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        weight_decay: float,
        lambda_mse: float,
        lambda_xent: float,
    ):
        super().__init__()

        self.feature_extrator = utils.freeze_model(feature_extractor)
        self.classification_head = utils.freeze_model(classification_head)
        self.teacher_module = utils.freeze_model(teacher_module)

        self.adapter = utils.freeze_model(adapter)

        # sanity check
        for module in [
            self.feature_extrator,
            self.classification_head,
            self.teacher_module,
        ]:
            _, trainable_param = utils.count_params_in_model(module)
            assert trainable_param == 0
            assert not module.training

        self.approximator = approximator

        self.lr = lr

        self.arr_metrics = []
        self.dataset = dataset

        self.lambda_mse = lambda_mse
        self.lambda_xent = lambda_xent
        print(f"Lambda (mse={self.lambda_mse}), (xent={self.lambda_xent})")

        # todo: find a better way to do this. Perhaps, via Callback?
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.weight_decay = weight_decay

        self.eval_safeguard()

    def configure_optimizers(self):
        # todo: log how many trainable params we have
        optimizer = torch.optim.Adam(
            self.approximator.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def forward_with_feats(
        self, feat
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            feat_in = self.feature_extrator(feat)

        feat_out = self.approximator(feat_in)

        approximated_act = self.adapter(feat_out)

        logits = self.classification_head(approximated_act)

        return feat_in, feat_out, logits

    def forward(self, feat) -> torch.Tensor:
        _, _, logits = self.forward_with_feats(feat)
        return logits

    def _compute_loss(self, batch, prefix):
        x, y = batch

        assert not self.feature_extrator.training
        assert not self.classification_head.training
        assert not self.teacher_module.training

        if prefix == "train":
            assert self.approximator.training

        feat_in, feat_out, logits = self.forward_with_feats(x)

        loss_xent = self._compute_xent_loss(logits, y)
        loss_mse = self._compute_mse_loss(feat_in, feat_out)
        loss = loss_xent + loss_mse

        self.log(f"{prefix}_loss_xent", loss_xent, on_epoch=True)
        self.log(f"{prefix}_loss_mse", loss_mse, on_epoch=True)
        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        return loss

    def _compute_xent_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = logits[:, self.dataset.selected_classes]
        ynew = self.dataset.transform_target(y)

        return self.lambda_xent * F.cross_entropy(logits, ynew)

    def _compute_mse_loss(
        self, feat_in: torch.Tensor, feat_out: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            expected_out = self.teacher_module(feat_in)

        _, _, h, w = expected_out.shape

        loss_mse = F.mse_loss(feat_out, expected_out, reduction="none")
        loss_mse = loss_mse.flatten(start_dim=1) / (h * w)
        loss_mse = loss_mse.sum(dim=1)

        return self.lambda_mse * loss_mse.mean()

    def training_step(self, train_batch, batch_idx):
        return self._compute_loss(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self._compute_loss(val_batch, "val")

    def eval_safeguard(self):
        self.feature_extrator.eval()
        self.classification_head.eval()
        self.teacher_module.eval()

    def on_fit_start(self) -> None:
        self.eval_safeguard()

    def on_train_batch_start(self, batch, batch_idx) -> int | None:
        status = super().on_train_batch_start(batch, batch_idx)

        self.eval_safeguard()

        return status

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
            self.teacher.to(device),
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
        log_dir: Path,
        lambda_mse: float,
        lambda_xent: float,
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
        _, teacher_module, _ = models.resnet.split_resnet_18_at(
            self.teacher, distill_info.layer_name
        )

        (
            feature_extractor,
            _,
            classification_head,
        ) = models.resnet.split_resnet_18_at(student, distill_info.layer_name)

        training_wrapper = ModelWrapper(
            feature_extractor=feature_extractor,
            teacher_module=nn.Sequential(
                teacher_module,
                basis.construct_adapter(
                    k=distill_info.num_output_channels,
                    device=device,
                    mode=bases.AdapterMode.ENCODER,
                ),
            ),
            adapter=basis.construct_adapter(
                k=distill_info.num_output_channels,
                device=device,
                mode=bases.AdapterMode.DECODER,
            ),
            approximator=approx_mod,
            classification_head=classification_head,
            lr=lr,
            dataset=self.dataset,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            weight_decay=self.weight_decay,
            lambda_mse=lambda_mse,
            lambda_xent=lambda_xent,
        )

        student.to(device)

        student_acc_before_training = metrics.accuracy_with_subclasses(
            nn.Sequential(
                feature_extractor,
                approx_mod,
                training_wrapper.adapter,
                classification_head,
            ),
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
