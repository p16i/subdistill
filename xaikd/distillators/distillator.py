import typing


from pytorch_lightning.loggers import Logger


import pytorch_lightning as pl
import numpy as np


import torch
from torch import nn
from torch.utils.data import DataLoader


from pathlib import Path


from xaikd import distillation_policies
from xaikd import datasets
from xaikd import metrics

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from .trainer_wrapper import LayerwiseKDModelWrapper


class Layerwise:
    def __init__(
        self,
        teacher: nn.Module,
        dataset: datasets.DatasetConfiguration,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str,
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
