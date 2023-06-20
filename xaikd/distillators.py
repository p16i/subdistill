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
    output_spatial_dims: typing.Tuple[int, int]


def get_distill_infor(
    arch: str, layer: str, compression_rate: float
) -> LayerDistillInfo:
    assert arch == "cifar100-resnet18-p1"

    info = dict(
        zip(
            ["layer3", "layer4"],
            [
                LayerDistillInfo(
                    layer_name="layer3",
                    num_input_channels=128,
                    num_output_channels=int(256 * compression_rate),
                    output_spatial_dims=(8, 8),
                ),
                LayerDistillInfo(
                    layer_name="layer4",
                    num_input_channels=256,
                    num_output_channels=int(512 * compression_rate),
                    output_spatial_dims=(4, 4),
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
        dataset: datasets.TwoClassesDataset,
        val_dataloader: DataLoader,
    ):
        super().__init__()

        self.feature_extrator = feature_extractor
        self.approximator = approximator
        self.classification_head = classification_head

        self.lr = lr

        self.arr_metrics = []
        self.dataset = dataset
        self.val_loss = torchmetrics.MeanMetric()
        self.val_dataloader = val_dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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

        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        logits = self(x)

        loss = F.cross_entropy(logits, y)
        self.val_loss.update(loss)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute())

    def on_train_epoch_end(self) -> None:
        self.approximator.eval()

        auroc, _ = metrics.auroc(
            self,
            self.val_dataloader,
            self.dataset.selected_classes,
            self.device,
            should_convert_auroc=True,
        )

        self.logger.experiment.add_scalar(
            "auroc", auroc, global_step=self.current_epoch
        )

        self.arr_metrics.append(auroc)

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
        dataset: datasets.TwoClassesDataset,
        device: str,
    ) -> None:
        pass
        self.dataset = dataset

        self.teacher = teacher

        self.device = device
        self.ref_auroc, _ = metrics.auroc(
            teacher,
            self.dataset.loader(train_split=False),
            classes=self.dataset.selected_classes,
            device=self.device,
            should_convert_auroc=True,
        )

    def distill(
        self,
        student: nn.Module,
        approx_mod: nn.Module,
        distill_info: LayerDistillInfo,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        basis: bases.Basis,
        device: str,
        lr: float,
        log_dir=Path,
        seed=1,
    ):
        os.makedirs(str(log_dir), exist_ok=True)
        # # todo: deep copy should not change any
        # student = copy.deepcopy(self.teacher)
        student.to(device)

        # student.eval()

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

        decoder = basis.contruct_rank_d_decoder(distill_info.num_output_channels)
        decoder.to(device)

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
            val_dataloader=val_dataloader,
        )

        student_auroc_before_training, _ = metrics.auroc(
            student,
            val_dataloader,
            classes=self.dataset.selected_classes,
            device=self.device,
            should_convert_auroc=True,
        )
        print(
            f"Student AUROC Before Training: {student_auroc_before_training:.4f} (teacher={self.ref_auroc:.4f})"
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
        trainer.fit(training_wrapper, train_dataloader, val_dataloader)

        arr_metrics = []

        for epoch, auroc in enumerate(training_wrapper.arr_metrics):
            arr_metrics.append(
                dict(
                    layer=distill_info.layer_name,
                    epoch=epoch,
                    epoch_auroc=auroc,
                    teacher_auroc=self.ref_auroc,
                    student_auroc_before_training=student_auroc_before_training,
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


# class FromScratch(Grafting):
#     def distill(
#         self,
#         epochs: int,
#         basis_name: str,
#         basis_dir: Path,
#         device: str,
#         lr: float,
#         seed=1,
#     ):
#         utils.deactivate_requires_grad(self.teacher)

#         # todo: deep copy should not change any
#         student = copy.deepcopy(self.teacher)
#         student.to(device)

#         utils.deactivate_requires_grad(student)

#         ref_auroc = metrics.auroc(
#             student,
#             self.dataset.loader(train_split=False),
#             attributors.LogOddEvidence(self.dataset.selected_classes, self.dataset),
#             self.device,
#         )
#         ref_auroc = np.max([ref_auroc, 1 - ref_auroc])

#         arr_distill_info = self.setup()

#         for info in arr_distill_info:
#             approxer = self.on_training_layer_start(student, info, device=device)

#         count_total_params, count_trainable_params = utils.count_params_in_model(
#             student
#         )

#         assert count_trainable_params > 0

#         print("-----------")
#         for param in student.children():
#             print(param)
#             _1, _2 = utils.count_params_in_model(param)
#             print(f"> total_params: {_1} (trainable {_2})")
#         print("-----------")

#         # Optimizers specified in the torch.optim package
#         optimizer = torch.optim.SGD(student.parameters(), lr=lr)

#         arr_metrics = []

#         tbar = tqdm(total=epochs)
#         steps = 1
#         for epoch in range(epochs):
#             for x, y in self.dataset.loader(train_split=True, shuffle=True):
#                 optimizer.zero_grad()
#                 logits = student(x.to(device))

#                 ybins = torch.where(y == self.dataset.selected_classes[0], 0, 1)
#                 loss = F.cross_entropy(
#                     logits[:, self.dataset.selected_classes], ybins.to(device)
#                 )

#                 loss.backward()

#                 optimizer.step()
#                 steps += 1
#                 log_value("loss", loss, steps)
#                 log_value("epoch", epoch, steps)

#             auroc = metrics.auroc(
#                 student,
#                 self.dataset.loader(train_split=False),
#                 attributors.LogOddEvidence(self.dataset.selected_classes, self.dataset),
#                 self.device,
#             )

#             auroc = np.max([auroc, 1 - auroc])

#             log_value("auroc", auroc, epoch)

#             tbar.update(1)
#             tbar.set_description(
#                 f"[AUROC={auroc:.4f} (teacher: {ref_auroc:.4f})| loss={float(loss.cpu().detach()):.4e}]"
#             )

#             arr_metrics.append(
#                 dict(
#                     layer="all",
#                     global_epoch=epochs,
#                     layer_epoch=epoch,
#                     auroc=auroc,
#                     teacher_auroc=ref_auroc,
#                 )
#             )

#         return arr_metrics

#     def on_training_layer_start(
#         self,
#         student: nn.Module,
#         distil_info: LayerDistillInfo,
#         device: str,
#     ):
#         if distil_info.layer_name == "layer4":
#             adapter = torch.nn.Conv2d(
#                 distil_info.num_output_channels,
#                 int(distil_info.num_output_channels * (1 / self.compression_rate)),
#                 kernel_size=1,
#             )
#         else:
#             adapter = torch.nn.Identity()
#         approx_mod = ApproximationModule(
#             adapter=adapter,
#             num_input_channels=distil_info.num_input_channels,
#             num_output_channels=distil_info.num_output_channels,
#             output_spatial_dims=distil_info.output_spatial_dims,
#         )

#         approx_mod.to(device)

#         setattr(student, distil_info.layer_name, approx_mod)

#         return approx_mod
