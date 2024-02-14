import typing
from typing import Any, Iterable, Optional


from pytorch_lightning.loggers import Logger


import pytorch_lightning as pl
import numpy as np


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


from pathlib import Path


from xaikd import distillation_policies, utils, datasets, bases, models
from xaikd.utils import metrics

from torchmetrics import Accuracy, MeanMetric

from pytorch_lightning.callbacks import LearningRateMonitor


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
        lr: float,
        lambda_layer: float,
        lambda_task: float,
        lambda_kd: float,
        num_classes: int,
    ):
        super().__init__()

        # We wrap teacher to a non-PyTorch Module class
        # in order to prevents Lightning to set the teacher to training mode.
        self.teacher = Teacher(teacher)

        self.student = student
        self.layer_policy_collection = layerwise_policies

        # ref: temperature value from
        # https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L78
        self.last_layer_policy = distillation_policies.KLPolicy(temperature=4.0)
        self.lr = lr

        self.arr_metrics = []

        self.lambda_layer = lambda_layer
        self.lambda_task = lambda_task
        self.lambda_kd = lambda_kd

        print(
            f"Lambda (task={self.lambda_task}, layer={self.lambda_layer}, logit={self.lambda_kd} )"
        )

        self.metric = dict(
            train_acc=Accuracy(task="multiclass", num_classes=num_classes),
            val_acc=Accuracy(task="multiclass", num_classes=num_classes),
            train_agreement=MeanMetric(),
            val_agreement=MeanMetric(),
            train_agreement_on_teacher_correct=MeanMetric(),
            val_agreement_on_teacher_correct=MeanMetric(),
        )

        self.arr_metrics = dict(
            train_acc=[],
            val_acc=[],
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

        # ref: https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L273
        # optimizer = torch.optim.SGD(
        #     parameters, lr=0.01, momentum=0.9, weight_decay=5e-4
        # )
        # return optimizer

    def _compute_loss(self, batch, prefix, batch_idx):
        x, y = batch

        with torch.no_grad():
            (
                teacher_logits,
                teacher_arr_intermediate_feats,
            ) = utils.interceptor.forward_and_intercept_intermediate_layers(
                self.teacher.model,
                x,
                layers=self.layer_policy_collection.teacher_layers,
            )

        (
            student_logits,
            student_arr_intermediate_feats,
        ) = utils.interceptor.forward_and_intercept_intermediate_layers(
            self.student,
            x,
            layers=self.layer_policy_collection.student_layers,
        )

        loss_task = F.cross_entropy(student_logits, y)
        loss_kd = self.last_layer_policy(teacher_logits, student_logits)

        loss_layer = 0
        for lix, policy in enumerate(self.layer_policy_collection.policies):
            _loss_layer = policy(
                teacher_arr_intermediate_feats[lix], student_arr_intermediate_feats[lix]
            )

            loss_layer += _loss_layer

            layer_name = self.layer_policy_collection.student_layers[lix]

            self.log(f"{prefix}_loss_layer_{layer_name}", _loss_layer, on_epoch=True)

            policy.transformer_teacher_feats

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

        loss = (
            self.lambda_task * loss_task
            + self.lambda_kd * loss_kd
            + self.lambda_layer * loss_layer
        )

        self.log(f"{prefix}_loss_task", loss_task, on_epoch=True)
        self.log(f"{prefix}_loss_kd", loss_kd, on_epoch=True)
        self.log(f"{prefix}_loss_layer", loss_layer, on_epoch=True)
        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        teacher_y_pred = torch.argmax(teacher_logits, dim=1).detach().cpu()
        student_y_pred = torch.argmax(student_logits, dim=1).detach().cpu()

        self.metric[f"{prefix}_acc"].update(student_y_pred, y.cpu())
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
        for suffix in ["acc", "agreement", "agreement_on_teacher_correct"]:
            slug = f"{prefix}_{suffix}"

            metric = self.metric[slug]
            value = metric.compute()
            metric.reset()

            if not self.trainer.sanity_checking:
                self.log(slug, value)
                self.arr_metrics[slug].append(float(value))

    def on_validation_epoch_end(self) -> None:
        self._compute_metric("val")

        for layer, policy in zip(
            self.layer_policy_collection.student_layers,
            self.layer_policy_collection.policies,
        ):
            if (
                hasattr(policy.transformer_student_feats, "weight")
                and policy.transformer_student_feats.weight is not None
            ):
                W = policy.transformer_student_feats.weight
                slug = f"student-transform-norm--{layer}"

                self.log(slug, torch.linalg.matrix_norm(W.squeeze()))

    def on_train_epoch_end(self) -> None:
        self._compute_metric("train")

    def on_save_checkpoint(self, checkpoint):
        return dict(student=self.student)


class Layerwise:
    def __init__(
        self,
        teacher: nn.Module,
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

        with torch.no_grad():
            self.ref_acc, self.ref_xent = metrics.accuracy(
                self.teacher.to(device),
                val_dataloader,
                num_classes=dataset.num_classes,
                device=self.device,
            )

        self.weight_decay = weight_decay

    def distill(
        self,
        student: nn.Module,
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
        # callbacks=[],
    ) -> typing.Tuple[nn.Module, typing.Dict]:
        student.to(device)

        with torch.no_grad():
            (
                student_acc_before_training,
                student_xent_before_training,
            ) = metrics.accuracy(
                student,
                dataloader=self.val_dataloader,
                num_classes=self.dataset.num_classes,
                device=self.device,
            )

        print(
            f"[before training] metrics: student (teacher) | acc={student_acc_before_training:.4f} ({self.ref_acc:.4f}), xent={student_xent_before_training:.4f} ({self.ref_xent:.4f})"
        )

        # we set the seed here again because to make sure that the state of random generator for
        # training is the same for all policies.
        # Said differently, some policies also contain random initialization of nn.Module
        # which then alter state of randomization.
        pl.seed_everything(seed)

        training_wrapper = LayerwiseKDModelWrapper(
            teacher=self.teacher,
            student=student,
            layerwise_policies=layer_policies,
            lr=lr,
            lambda_task=lambda_task,
            lambda_kd=lambda_kd,
            lambda_layer=lambda_layer,
            num_classes=self.dataset.num_classes,
        )

        print(f"Training log is saved to `{log_dir}`")

        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=epochs,
            logger=logger,
            log_every_n_steps=1,
            enable_checkpointing=enable_checkpointing,
            deterministic="warn",
            callbacks=[LearningRateMonitor(logging_interval="step")],
        )

        trainer.fit(training_wrapper, self.train_dataloader, self.val_dataloader)

        self.post_training_sanitycheck(
            student=student,
            device=device,
            expected_acc=training_wrapper.arr_metrics["val_acc"][-1],
        )

        experiment_stat = dict(
            teacher_acc=self.ref_acc,
            student_acc_before_training=student_acc_before_training,
            arr_metrics=training_wrapper.arr_metrics,
        )

        return student, experiment_stat

    def post_training_sanitycheck(
        self,
        student: nn.Module,
        expected_acc: float,
        device: str,
    ):
        student.to(device)
        # sanity check: acc from student to should equal to the one we have evaluated!
        with torch.no_grad():
            actual_acc, _ = metrics.accuracy(
                student,
                self.val_dataloader,
                num_classes=self.dataset.num_classes,
                device=device,
            )

            np.testing.assert_allclose(
                actual_acc,
                expected_acc,
                err_msg="accuracy computed from modified student should match the last one returned from distillator",
            )
