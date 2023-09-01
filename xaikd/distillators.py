import copy
import os
import typing


from pytorch_lightning.loggers import TensorBoardLogger, Logger


import pytorch_lightning as pl
import numpy as np


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


from pathlib import Path


from xaikd import utils, datasets, bases, models, criteria
from xaikd.utils import metrics
from xaikd.distillation_info import LayerDistillInfo

from torchmetrics import Accuracy

from pytorch_lightning.callbacks import LearningRateMonitor


class LayerwiseKDModelWrapper(pl.LightningModule):
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        layerwise_policies: typing.List[typing.Tuple[str, nn.Module]],
        lr: float,
        lambda_layer: float,
        lambda_task: float,
        lambda_kd: float,
        num_classes: int,
    ):
        super().__init__()

        self.teacher = utils.freeze_model(teacher)
        self.student = student
        self.policies = layerwise_policies

        self.layer_names = list(map(lambda t: t[0], layerwise_policies))

        # ref: temperature value from
        # https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L78
        self.last_layer_policy = criteria.KL(temperature=4.0)

        # sanity check
        for module in [
            self.teacher,
        ]:
            _, trainable_param = utils.count_params_in_model(module)
            assert trainable_param == 0
            assert not module.training

        self.lr = lr

        self.arr_metrics = []

        self.lambda_layer = lambda_layer
        self.lambda_task = lambda_task
        self.lambda_kd = lambda_kd

        print(
            f"Lambda (task={self.lambda_task}, layer={self.lambda_layer}, logit={self.lambda_kd} )"
        )

        self.eval_safeguard()

        self.metric = dict(
            train=Accuracy(task="multiclass", num_classes=num_classes),
            val=Accuracy(task="multiclass", num_classes=num_classes),
        )

        self.arr_metrics = dict(train=[], val=[])

    def configure_optimizers(self):
        parameters = list(self.student.parameters())

        # get parameters from student and transformation in criteria
        for layer, criteria in self.policies:
            parameters.extend(criteria.parameters())

        # previous optimizer
        optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [scheduler]

        # ref: https://github.com/HobbitLong/RepDistiller/blob/dcc043277f2820efafd679ffb82b8e8195b7e222/train_student.py#L273
        # optimizer = torch.optim.SGD(
        #     parameters, lr=0.01, momentum=0.9, weight_decay=5e-4
        # )
        # return optimizer

    def eval_safeguard(self):
        self.teacher.eval()

    def on_fit_start(self) -> None:
        self.eval_safeguard()

    def on_train_batch_start(self, batch, batch_idx) -> typing.Union[int, None]:
        status = super().on_train_batch_start(batch, batch_idx)

        self.eval_safeguard()

        return status

    def _compute_loss(self, batch, prefix, batch_idx):
        x, y = batch

        assert not self.teacher.training

        with torch.no_grad():
            (
                teacher_logits,
                teacher_arr_intermediate_feats,
            ) = utils.interceptor.forward_and_intercept_intermediate_layers(
                self.teacher,
                x,
                layers=self.layer_names,
            )

        (
            student_logits,
            student_arr_intermediate_feats,
        ) = utils.interceptor.forward_and_intercept_intermediate_layers(
            self.student,
            x,
            layers=self.layer_names,
        )

        loss_task = self.lambda_task * F.cross_entropy(student_logits, y)
        loss_kd = self.lambda_kd * self.last_layer_policy(
            teacher_logits, student_logits
        )

        loss_layer = 0
        for lix, (layer, policy) in enumerate(self.policies):
            loss_layer += self.lambda_layer * policy(
                teacher_arr_intermediate_feats[lix], student_arr_intermediate_feats[lix]
            )

        loss = loss_task + loss_kd + loss_layer

        self.log(f"{prefix}_loss_task", loss_task, on_epoch=True)
        self.log(f"{prefix}_loss_kd", loss_kd, on_epoch=True)
        self.log(f"{prefix}_loss_layer", loss_layer, on_epoch=True)
        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        self.metric[prefix].update(
            torch.argmax(student_logits, dim=1).detach().cpu(), y.cpu()
        )

        return loss

    def training_step(self, train_batch, batch_idx):
        return self._compute_loss(train_batch, "train", batch_idx)

    def validation_step(self, val_batch, batch_idx):
        return self._compute_loss(val_batch, "val", batch_idx)

    def _compute_metric(self, prefix):
        metric = self.metric[prefix]
        val = metric.compute()
        metric.reset()
        self.log(f"{prefix}_acc", val)
        self.arr_metrics[prefix].append(float(val))

    def on_validation_epoch_end(self) -> None:
        self._compute_metric("val")

    def on_train_epoch_end(self) -> None:
        self._compute_metric("train")

    def on_save_checkpoint(self, checkpoint):
        raise NotImplemented("to be update; we should only save student")
        checkpoint["approximator"] = self.approximator
        checkpoint["adapter"] = self.adapter
        return checkpoint


class Layerwise:
    def __init__(
        self,
        teacher: models.interfaces.DistillableModel,
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
        layer_policies: typing.List[typing.Tuple[str, nn.Module]],
        epochs: int,
        device: str,
        lr: float,
        log_dir: Path,
        logger: Logger,
        lambda_task: float,
        lambda_kd: float,
        lambda_layer: float,
        # enable_checkpointing=False,
        # callbacks=[],
    ) -> typing.Tuple[nn.Module, typing.Dict]:
        student.to(device)

        (
            student_acc_before_training,
            student_xent_before_training,
        ) = metrics.accuracy(
            student,
            dl=self.val_dataloader,
            num_classes=self.dataset.num_classes,
            device=self.device,
        )

        print(
            f"[before training] metrics: student (teacher) | acc={student_acc_before_training:.4f} ({self.ref_acc:.4f}), xent={student_xent_before_training:.4f} ({self.ref_xent:.4f})"
        )

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
            enable_checkpointing=False,
            deterministic=True,
            callbacks=[LearningRateMonitor(logging_interval="step")],
        )

        trainer.fit(training_wrapper, self.train_dataloader, self.val_dataloader)

        # todo: to activate
        # self.finalize_and_verify_student_with_adapter(
        #     student=student,
        #     distill_info=distill_info,
        #     approximator=approximator,
        #     adapter=training_wrapper.adapter,
        #     device=device,
        #     expected_acc=training_wrapper.arr_metrics["val"][-1],
        # )

        experiment_stat = dict(
            teacher_acc=self.ref_acc,
            student_acc_before_training=student_acc_before_training,
            arr_metrics=training_wrapper.arr_metrics,
        )

        return student, experiment_stat

    # def setup_student_with_approximator(
    #     self,
    #     student: nn.Module,
    #     approximator: nn.Module,
    #     distill_info: LayerDistillInfo,
    # ):
    #     utils.freeze_model(student)

    #     approx_total_params, approx_learnable_params = utils.count_params_in_model(
    #         approximator
    #     )

    #     replaced_module_total_params, _ = utils.count_params_in_model(
    #         getattr(student, distill_info.layer_name)
    #     )

    #     before_total_params, _ = utils.count_params_in_model(student)

    #     # this is the actual logic of the function the rest simply for assertions!
    #     setattr(student, distill_info.layer_name, approximator)

    #     after_total_params, after_learnable_params = utils.count_params_in_model(
    #         student
    #     )

    #     np.testing.assert_equal(after_learnable_params, approx_learnable_params)
    #     np.testing.assert_equal(
    #         before_total_params - replaced_module_total_params + approx_total_params,
    #         after_total_params,
    #     )

    def finalize_and_verify_student_with_adapter(
        self,
        student: nn.Module,
        distill_info: LayerDistillInfo,
        approximator: nn.Module,
        adapter: nn.Module,
        expected_acc: float,
        device: str,
    ):
        assert getattr(student, distill_info.layer_name) == approximator

        setattr(
            student,
            distill_info.layer_name,
            torch.nn.Sequential(
                approximator,
                adapter,
            ),
        )
        student.eval()
        student.to(device)

        # sanity check: acc from student to should equal to the one we have evaluated!
        with torch.no_grad():
            actual_acc, _ = metrics.accuracy_with_subclasses(
                student,
                self.val_dataloader,
                considered_classes=self.dataset.selected_classes,
                transform_target=self.dataset.transform_target,
                device=device,
            )
            np.testing.assert_allclose(
                actual_acc,
                expected_acc,
                err_msg="accuracy computed from modified student should match the last one returned from distillator",
            )
