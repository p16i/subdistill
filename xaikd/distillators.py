import typing
from typing import Any, Iterable, Optional


from pytorch_lightning.loggers import Logger


import pytorch_lightning as pl
import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


from pathlib import Path


from xaikd import distillation_policies, utils, bases, models
from xaikd import datasets
from xaikd import metrics

from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


def should_detach_output(partition_mode: str, current_epoch: int) -> bool:
    # partition_mode = @<int>
    _, expected_epoch = partition_mode.split("@")
    expected_epoch = int(expected_epoch)

    output = current_epoch < expected_epoch

    return output


class Teacher(object):
    """The class is a wrapper to a PyTorch model.
    Its purpose is to prevent Lightning to set the wrapped model to training mode.

    Args:
        model (nn.Module):
    """

    def __init__(self, model: torch.nn.Module):
        assert not model.training
        self.model = utils.freeze_model(model)

    def __call__(self, *args: Any) -> torch.Tensor:
        return self.model(*args)


class LayerwiseKDModelWrapper(pl.LightningModule):
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        layerwise_policies: distillation_policies.LayerPolicyCollection,
        last_layer_policy: str,
        lr: float,
        lambda_layer: float,
        lambda_task: float,
        lambda_kd: float,
        num_classes: int,
        parameter_partition_mode: str,
        finetuning_with_layer_loss: bool,
    ):
        super().__init__()

        # We wrap teacher to a non-PyTorch Module class
        # in order to prevents Lightning to set the teacher to training mode.
        self.teacher = Teacher(teacher)

        self.student = student
        self.layer_policy_collection = layerwise_policies

        self.last_layer_policy: distillation_policies.LastLayerPolicy = (
            distillation_policies.get_last_layer_policy(last_layer_policy)
        )
        self.lr = lr

        self.arr_metrics = []

        self.lambda_layer = lambda_layer
        self.lambda_task = lambda_task
        self.lambda_kd = lambda_kd
        self.parameter_partition_mode = parameter_partition_mode
        self.finetuning_with_layer_loss = finetuning_with_layer_loss

        print(
            f"Lambda (task={self.lambda_task}, layer={self.lambda_layer}, logit={self.lambda_kd} )"
        )

        # todo: perhaps, torchvision has some functionairty for tihs
        self.metric = dict(
            train_auroc=BinaryAUROC(thresholds=100),
            val_auroc=BinaryAUROC(thresholds=100),
            train_agreement=MeanMetric(),
            val_agreement=MeanMetric(),
            train_agreement_on_teacher_correct=MeanMetric(),
            val_agreement_on_teacher_correct=MeanMetric(),
        )

        self.arr_metrics = dict(
            train_auroc=[],
            val_auroc=[],
            train_agreement=[],
            val_agreement=[],
            train_agreement_on_teacher_correct=[],
            val_agreement_on_teacher_correct=[],
        )

    def _get_parameters(self) -> typing.List[nn.Parameter]:
        # get parameters from student and transformation in criteria
        parameters = list(self.student.parameters())

        parameters = parameters + list(self.layer_policy_collection.parameters())

        return parameters

    def configure_optimizers(self):
        parameters = self._get_parameters()

        # pat's optimizer (used in S11, 12)
        optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [scheduler]

        # todo:  remove this
        # ref: https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L273
        # optimizer = torch.optim.SGD(
        #     parameters, lr=0.01, momentum=0.9, weight_decay=5e-4
        # )
        # return optimizer

    def _compute_loss(self, batch, prefix, batch_idx):
        x, y = batch
        n = x.shape[0]
        if prefix == "val":
            assert not self.student.training

        with torch.no_grad():
            (
                teacher_logits,
                teacher_arr_intermediate_feats,
            ) = utils.interceptor.forward_and_intercept_intermediate_layers(
                self.teacher.model,
                x,
                layers=self.layer_policy_collection.teacher_layers,
                detach_output=False,
            )
        (
            student_logits,
            student_arr_intermediate_feats,
        ) = utils.interceptor.forward_and_intercept_intermediate_layers(
            self.student,
            x,
            layers=self.layer_policy_collection.student_layers,
            detach_output=should_detach_output(
                self.parameter_partition_mode, self.current_epoch
            ),
        )

        assert student_logits.shape == (n, 1)

        student_logits = student_logits.squeeze(1)

        loss_task = F.binary_cross_entropy_with_logits(student_logits, y.float())

        loss_kd = self.last_layer_policy(teacher_logits, student_logits, y)

        loss_layer = torch.tensor(0.0).to(loss_kd.device)

        is_finetuning = not should_detach_output(
            partition_mode=self.parameter_partition_mode,
            current_epoch=self.current_epoch,
        )

        if is_finetuning and not self.finetuning_with_layer_loss:
            layer_policies = []
        else:
            layer_policies = self.layer_policy_collection.policies

        for lix, policy in enumerate(layer_policies):

            _loss_layer = policy(
                teacher_arr_intermediate_feats[lix], student_arr_intermediate_feats[lix]
            )

            loss_layer = loss_layer + _loss_layer

            layer_name = self.layer_policy_collection.student_layers[lix]

            self.log(f"{prefix}_loss_layer_{layer_name}", _loss_layer, on_epoch=True)
            if prefix == "val":
                for label, act in (
                    ("student", student_arr_intermediate_feats[lix]),
                    (
                        "teacher",
                        policy.transformer_teacher_feats(
                            teacher_arr_intermediate_feats[lix]
                        ),
                    ),
                ):
                    norm = torch.linalg.norm(act, dim=1)
                    layer = self.layer_policy_collection.student_layers[lix]

                    self.log(
                        f"{prefix}_actnorm_{label}_{layer}_min",
                        norm.min(),
                        on_epoch=True,
                    )
                    self.log(
                        f"{prefix}_actnorm_{label}_{layer}_max",
                        norm.max(),
                        on_epoch=True,
                    )
                    self.log(
                        f"{prefix}_actnorm_{label}_{layer}_mean",
                        norm.mean(),
                        on_epoch=True,
                    )
                    self.log(
                        f"{prefix}_actnorm_{label}_{layer}_median",
                        norm.median(),
                        on_epoch=True,
                    )

        is_shown_in_prog_bar = prefix == "val"
        loss = 0

        if self.lambda_task > 0:
            assert torch.isfinite(loss_task)
            loss = loss + self.lambda_task * loss_task
            self.log(
                f"{prefix}_loss_task",
                loss_task,
                on_epoch=True,
                prog_bar=is_shown_in_prog_bar,
            )
        if self.lambda_kd > 0:
            assert torch.isfinite(loss_kd)
            loss = loss + self.lambda_kd * loss_kd
            self.log(
                f"{prefix}_loss_kd",
                loss_kd,
                on_epoch=True,
                prog_bar=is_shown_in_prog_bar,
            )
        if self.lambda_layer > 0:
            assert torch.isfinite(loss_layer)
            loss = loss + self.lambda_layer * loss_layer

            self.log(
                f"{prefix}_loss_layer",
                loss_layer,
                on_epoch=True,
                prog_bar=is_shown_in_prog_bar,
            )
        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        teacher_y_pred = (teacher_logits > 0).detach().cpu()
        student_y_pred = (student_logits > 0).detach().cpu()

        self.metric[f"{prefix}_auroc"].update(student_logits.detach().cpu(), y.cpu())
        self.metric[f"{prefix}_agreement"].update(student_y_pred == teacher_y_pred)
        self.metric[f"{prefix}_agreement_on_teacher_correct"].update(
            (teacher_y_pred == y.cpu()) * (student_y_pred == teacher_y_pred)
        )

        return loss

    def training_step(self, train_batch, batch_idx):
        return self._compute_loss(train_batch, "train", batch_idx)

    def validation_step(self, val_batch, batch_idx):
        return self._compute_loss(val_batch, "val", batch_idx)

    def _compute_metric(self, prefix):
        for suffix in ["auroc", "agreement", "agreement_on_teacher_correct"]:
            slug = f"{prefix}_{suffix}"

            metric = self.metric[slug]
            value = metric.compute()

            if suffix == "auroc":
                value = np.max([value, 1 - value])

            metric.reset()

            if not self.trainer.sanity_checking:
                self.log(slug, value)
                self.arr_metrics[slug].append(float(value))

    def on_validation_epoch_end(self) -> None:
        self._compute_metric("val")

        # todo: check whether we still need to log this
        # for layer, policy in zip(
        #     self.layer_policy_collection.student_layers,
        #     self.layer_policy_collection.policies,
        # ):
        #     if (
        #         hasattr(policy.transformer_student_feats, "weight")
        #         and policy.transformer_student_feats.weight is not None
        #     ):
        #         W = policy.transformer_student_feats.weight
        #         slug = f"student-transform-norm--{layer}"

        #         self.log(slug, torch.linalg.matrix_norm(W.squeeze()))

    def on_train_epoch_end(self) -> None:
        self._compute_metric("train")


class Layerwise:
    def __init__(
        self,
        teacher: nn.Module,
        dataset: datasets.DatasetConfiguration,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str,
        weight_decay: float,
        parameter_partition_mode: str,
    ) -> None:
        self.dataset = dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.teacher = teacher

        self.device = device

        self.metric_func = metrics.MetricAUROCBinaryCrossEntropy()

        with torch.no_grad():
            self.ref_auroc, self.ref_xent = self.metric_func(
                self.teacher.to(device),
                val_dataloader,
                device=self.device,
            )

        self.weight_decay = weight_decay
        self.parameter_partition_mode = parameter_partition_mode

    def distill(
        self,
        student: nn.Module,
        last_layer_policy: str,
        layer_policies: distillation_policies.LayerPolicyCollection,
        epochs: int,
        device: str,
        lr: float,
        log_dir: Path,
        logger: Logger,
        lambda_task: float,
        lambda_kd: float,
        lambda_layer: float,
        seed: int,
        enable_checkpointing: bool,
        finetuning_with_layer_loss: bool,
        # callbacks=[],
    ) -> typing.Tuple[nn.Module, typing.Dict]:

        assert (np.array([lambda_task, lambda_kd, lambda_layer]) > 0).any()

        student.eval()
        student.to(device)

        with torch.no_grad():
            (
                student_auroc_before_training,
                student_xent_before_training,
            ) = self.metric_func(
                student,
                dataloader=self.val_dataloader,
                device=self.device,
            )

        print(
            f"[before training] metrics: student (teacher) | auroc={student_auroc_before_training:.4f} ({self.ref_auroc:.4f}), xent={student_xent_before_training:.4f} ({self.ref_xent:.4f})"
        )

        # we set the seed here again because to make sure that the state of random generator for
        # training is the same for all policies.
        # Said differently, some policies also contain random initialization of nn.Module
        # which then alter state of randomization.
        pl.seed_everything(seed)

        training_wrapper = LayerwiseKDModelWrapper(
            teacher=self.teacher,
            student=student,
            last_layer_policy=last_layer_policy,
            layerwise_policies=layer_policies,
            lr=lr,
            lambda_task=lambda_task,
            lambda_kd=lambda_kd,
            lambda_layer=lambda_layer,
            num_classes=self.dataset.num_classes,
            parameter_partition_mode=self.parameter_partition_mode,
            finetuning_with_layer_loss=finetuning_with_layer_loss,
        )

        print(f"Training log is saved to `{log_dir}`")

        callback_checkpoint = (
            [
                ModelCheckpoint(
                    every_n_epochs=epochs // 2
                ),  # here, we save two checkpoints; middle and last epochs
            ]
            if enable_checkpointing
            else []
        )

        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=epochs,
            logger=logger,
            log_every_n_steps=1,
            deterministic="warn",
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                *callback_checkpoint,
            ],
        )

        trainer.fit(training_wrapper, self.train_dataloader, self.val_dataloader)

        student.eval()

        self.post_training_sanitycheck(
            student=student,
            device=device,
            expected=training_wrapper.arr_metrics["val_auroc"][-1],
        )

        experiment_stat = dict(
            teacher_auroc=self.ref_auroc,
            student_auroc_before_training=student_auroc_before_training,
            arr_metrics=training_wrapper.arr_metrics,
        )

        assert not student.training

        return student, experiment_stat

    def post_training_sanitycheck(
        self,
        student: nn.Module,
        expected: float,
        device: str,
    ):
        student.to(device)
        # sanity check: acc from student to should equal to the one we have evaluated!
        with torch.no_grad():

            actual, _ = self.metric_func(
                student,
                self.val_dataloader,
                device=device,
            )

            np.testing.assert_allclose(
                actual,
                expected,
                err_msg="stats computed from modified student should match the last one returned from distillator",
            )
