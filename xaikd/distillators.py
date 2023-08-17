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


from xaikd import utils, datasets, bases, models
from xaikd.utils import metrics
from xaikd.distillation_info import LayerDistillInfo

from torchmetrics import Accuracy

from pytorch_lightning.callbacks import LearningRateMonitor


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
        self.weight_decay = weight_decay

        self.eval_safeguard()

        self.metric = dict(
            train=Accuracy(task="multiclass", num_classes=dataset.num_classes),
            val=Accuracy(task="multiclass", num_classes=dataset.num_classes),
        )

        self.arr_metrics = dict(train=[], val=[])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.approximator.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        return [optimizer], [scheduler]

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

    def eval_safeguard(self):
        self.feature_extrator.eval()
        self.classification_head.eval()
        self.teacher_module.eval()

    def on_fit_start(self) -> None:
        self.eval_safeguard()

    def on_train_batch_start(self, batch, batch_idx) -> typing.Union[int, None]:
        status = super().on_train_batch_start(batch, batch_idx)

        self.eval_safeguard()

        return status

    def _compute_loss(self, batch, prefix, batch_idx):
        x, y = batch

        assert not self.feature_extrator.training
        assert not self.classification_head.training
        assert not self.teacher_module.training

        if prefix == "train":
            assert self.approximator.training
        else:
            assert not self.approximator.training

        feat_in, feat_out, logits = self.forward_with_feats(x)

        # remark: here we transform `y` (from original dataset) to a new index set
        selected_logits = logits[:, self.dataset.selected_classes]
        transformed_y = self.dataset.transform_target(y)

        loss_xent = self._compute_xent_loss(selected_logits, transformed_y)
        loss_mse = self._compute_mse_loss(feat_in, feat_out, batch_idx, prefix)
        loss = loss_xent + loss_mse

        self.log(f"{prefix}_loss_xent", loss_xent, on_epoch=True)
        self.log(f"{prefix}_loss_mse", loss_mse, on_epoch=True)
        self.log(f"{prefix}_loss_all", loss, on_epoch=True)

        self.metric[prefix].update(
            torch.argmax(selected_logits, dim=1).detach().cpu(),
            transformed_y.detach().cpu(),
        )

        return loss

    def _compute_xent_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.lambda_xent * F.cross_entropy(logits, y)

    def _compute_mse_loss(
        self, feat_in: torch.Tensor, feat_out: torch.Tensor, batch_idx, prefix
    ) -> torch.Tensor:
        with torch.no_grad():
            expected_out = self.teacher_module(feat_in)

        _, _, w, h = expected_out.shape

        loss_mse = F.mse_loss(feat_out, expected_out, reduction="none") / (w * h)
        loss_mse = loss_mse.flatten(start_dim=1)

        loss_mse = loss_mse.sum(dim=1)

        return self.lambda_mse * loss_mse.mean()

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

        self.ref_acc, self.ref_xent = metrics.accuracy_with_subclasses(
            self.teacher.to(device),
            val_dataloader,
            considered_classes=self.dataset.selected_classes,
            transform_target=self.dataset.transform_target,
            device=self.device,
        )

        self.weight_decay = weight_decay

    def distill(
        self,
        student: models.interfaces.DistillableModel,
        approximator: nn.Module,
        distill_info: LayerDistillInfo,
        epochs: int,
        basis: bases.Basis,
        device: str,
        lr: float,
        log_dir: Path,
        logger: Logger,
        lambda_mse: float,
        lambda_xent: float,
    ) -> typing.Tuple[nn.Module, typing.Dict]:
        os.makedirs(str(log_dir), exist_ok=True)

        print(f"Distilling layer={distill_info.layer_name} with {epochs} epochs")

        (
            total_teacher_params,
            _,
        ) = utils.count_params_in_model(self.teacher)

        self.setup_student_with_approximator(
            student, approximator=approximator, distill_info=distill_info
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
        # todo: this split should be available via the model itself.
        # e.g., self.teacher.split_at(...)
        _, teacher_module, _ = self.teacher.split_at(distill_info.layer_name)

        (
            feature_extractor,
            _,
            classification_head,
        ) = student.split_at(distill_info.layer_name)

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
            approximator=approximator,
            classification_head=classification_head,
            lr=lr,
            dataset=self.dataset,
            weight_decay=self.weight_decay,
            lambda_mse=lambda_mse,
            lambda_xent=lambda_xent,
        )

        student.to(device)

        (
            student_acc_before_training,
            student_xent_before_training,
        ) = metrics.accuracy_with_subclasses(
            nn.Sequential(
                feature_extractor,
                approximator,
                training_wrapper.adapter,
                classification_head,
            ),
            dl=self.val_dataloader,
            considered_classes=self.dataset.selected_classes,
            transform_target=self.dataset.transform_target,
            device=self.device,
        )

        print(
            f"[before training] metrics: student (teacher) | acc={student_acc_before_training:.4f} ({self.ref_acc:.4f}), xent={student_xent_before_training:.4f} ({self.ref_xent:.4f})"
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

        self.finalize_and_verify_student_with_adapter(
            student=student,
            distill_info=distill_info,
            approximator=approximator,
            adapter=training_wrapper.adapter,
            device=device,
            expected_acc=training_wrapper.arr_metrics["val"][-1],
        )

        experiment_stat = dict(
            layer=distill_info.layer_name,
            teacher_acc=self.ref_acc,
            student_acc_before_training=student_acc_before_training,
            student_trainable_param=count_trainable_params,
            student_total_params=count_total_params,
            teacher_total_params=total_teacher_params,
            arr_metrics=training_wrapper.arr_metrics,
        )

        return student, experiment_stat

    def setup_student_with_approximator(
        self,
        student: nn.Module,
        approximator: nn.Module,
        distill_info: LayerDistillInfo,
    ):
        utils.freeze_model(student)

        approx_total_params, approx_learnable_params = utils.count_params_in_model(
            approximator
        )

        replaced_module_total_params, _ = utils.count_params_in_model(
            getattr(student, distill_info.layer_name)
        )

        before_total_params, _ = utils.count_params_in_model(student)

        # this is the actual logic of the function the rest simply for assertions!
        setattr(student, distill_info.layer_name, approximator)

        after_total_params, after_learnable_params = utils.count_params_in_model(
            student
        )

        np.testing.assert_equal(after_learnable_params, approx_learnable_params)
        np.testing.assert_equal(
            before_total_params - replaced_module_total_params + approx_total_params,
            after_total_params,
        )

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
