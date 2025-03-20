import typing
from typing import Any

import pytorch_lightning as pl
import numpy as np


import torch
from torch import nn
from torch.nn import functional as F


from xaikd import distillation_policies, utils

from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAUROC


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

        self.lambda_layer = lambda_layer
        self.lambda_task = lambda_task
        self.lambda_kd = lambda_kd

        print(
            f"Lambda (task={self.lambda_task}, layer={self.lambda_layer}, logit={self.lambda_kd} )"
        )

        # todo: perhaps, torchvision has some functionality for tihs
        self.metric = dict(
            train_auroc=BinaryAUROC(thresholds=100),
            val_auroc=BinaryAUROC(thresholds=100),
        )

        # fixme: remove this
        self.arr_metrics = dict(
            train_auroc=[],
            val_auroc=[],
        )

    def _get_parameters(self) -> typing.List[nn.Parameter]:
        # get parameters from student and transformation in criteria
        parameters = list(self.student.parameters())

        parameters = parameters + list(self.layer_policy_collection.parameters())

        return parameters

    def configure_optimizers(self):
        parameters = self._get_parameters()

        optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
        return [optimizer], [scheduler]

    def _compute_loss_task(
        self, student_logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(student_logits, target.float())

    def _compute_loss_kd(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        loss = self.last_layer_policy(teacher_logits, student_logits, target)

        assert torch.isfinite(loss)

        return loss

    def _compute_loss_layer(
        self,
        teacher_arr_intermediate_feats: typing.List[torch.Tensor],
        student_arr_intermediate_feats: typing.List[torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:

        device = teacher_arr_intermediate_feats[0].device

        loss_layer = torch.tensor(0.0).to(device)

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

        assert torch.isfinite(loss_layer)

        return loss_layer

    def _compute_loss(self, batch, prefix, batch_idx):
        x, y = batch
        n = x.shape[0]

        assert not self.teacher.model.training

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
            detach_output=False,
        )

        assert student_logits.shape == (n, 1)

        student_logits = student_logits.squeeze(1)

        loss_task = self._compute_loss_task(student_logits=student_logits, target=y)
        loss_kd = self._compute_loss_kd(
            teacher_logits=teacher_logits, student_logits=student_logits, target=y
        )
        loss_layer = self._compute_loss_layer(
            teacher_arr_intermediate_feats=teacher_arr_intermediate_feats,
            student_arr_intermediate_feats=student_arr_intermediate_feats,
            prefix=prefix,
        )

        loss = torch.tensor(0.0).to(teacher_logits.device)

        for loss_label, loss_value, loss_coeff in [
            ("task", loss_task, self.lambda_task),
            ("kd", loss_kd, self.lambda_kd),
            ("layer", loss_layer, self.lambda_layer),
        ]:
            if loss_coeff == 0:
                continue

            assert torch.isfinite(loss_value)

            loss = loss + loss_coeff * loss_value

            self.log(
                f"{prefix}_loss_{loss_label}",
                loss_value,
                on_epoch=True,
                prog_bar=self._in_prog_bar(prefix),
            )

        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        self.metric[f"{prefix}_auroc"].update(student_logits.detach().cpu(), y.cpu())

        return loss

    def _in_prog_bar(self, prefix: str) -> bool:
        return prefix == "val"

    def training_step(self, train_batch, batch_idx):
        return self._compute_loss(train_batch, "train", batch_idx)

    def validation_step(self, val_batch, batch_idx):
        return self._compute_loss(val_batch, "val", batch_idx)

    def _compute_metric(self, prefix):
        for suffix in ["auroc"]:
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

        if len(self.arr_metrics["val_auroc"]) > 0:
            best_epoch = int(np.argmax(self.arr_metrics["val_auroc"]))
            best_val_auroc = float(self.arr_metrics["val_auroc"][best_epoch])
            self.log("best_epoch", best_epoch)
            # todo: log the value also on prog_bar
            self.log("best_val_auroc", best_val_auroc)

    def on_train_epoch_end(self) -> None:
        self._compute_metric("train")
