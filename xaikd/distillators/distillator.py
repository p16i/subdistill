import typing


from pytorch_lightning.loggers.wandb import WandbLogger
from wandb import Artifact
from wandb.wandb_run import Run


import pytorch_lightning as pl
import numpy as np


import torch
from torch import nn
from torch.utils.data import DataLoader


from pathlib import Path


from xaikd import distillation_policies, utils
from xaikd import datasets
from xaikd import metrics

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from .trainer_wrapper import LayerwiseKDModelWrapper


class Layerwise:
    def __init__(
        self,
        teacher: nn.Module,
        dataset: datasets.DatasetConfiguration,
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        dataloader_test: DataLoader,
        device: str,
    ) -> None:
        self.dataset = dataset
        self.dl_train = dataloader_train
        self.dl_val = dataloader_val
        self.dl_test = dataloader_test

        self.teacher = teacher

        self.device = device

        self.metric_func = metrics.MetricAUROCBinaryCrossEntropy()

        with torch.no_grad():
            self.ref_auroc, self.ref_xent = self.metric_func(
                self.teacher.to(device),
                dataloader_val,
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
        logger: WandbLogger,
        lambda_task: float,
        lambda_kd: float,
        lambda_layer: float,
        seed: int,
        upload_best_checkpoint: bool,
    ) -> nn.Module:

        assert (np.array([lambda_task, lambda_kd, lambda_layer]) > 0).any()

        student.eval()
        student.to(device)

        with torch.no_grad():
            (
                student_auroc_before_training,
                student_xent_before_training,
            ) = self.metric_func(
                student,
                dataloader=self.dl_val,
                device=self.device,
            )

        logger.experiment.summary["student_val_auroc_before_training"] = (
            student_auroc_before_training
        )
        logger.experiment.summary["teacher_auroc"] = self.ref_auroc

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

        callback_checkpoint = ModelCheckpoint(
            monitor="val_auroc",
            mode="max",
        )

        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=epochs,
            logger=logger,
            log_every_n_steps=1,
            deterministic="warn",
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                callback_checkpoint,
            ],
        )

        trainer.fit(training_wrapper, self.dl_train, self.dl_val)

        best_model_path = callback_checkpoint.best_model_path

        assert callback_checkpoint.best_model_score is not None

        best_epoch = np.argmax(training_wrapper.arr_metrics["val_auroc"])

        best_student = utils.modules.load_model_from_checkpoint(
            model_template_object=training_wrapper.student,
            checkpoint_path=best_model_path,
            model_key="student",
            device=device,
        )
        best_student.eval()

        self.post_training_sanitycheck(
            student=best_student,
            trainer=training_wrapper,
            checkpoint_callback=callback_checkpoint,
            device=device,
        )

        logger.experiment.summary["best_epoch"] = best_epoch

        logger.experiment.summary["student_best_val_auroc"] = (
            callback_checkpoint.best_model_score
        )
        self.log_test_metrics(best_student=best_student, logger=logger, device=device)

        if upload_best_checkpoint:
            wandb_run = logger.experiment
            self.log_model(
                wandb_run=wandb_run,
                artifact_name=f"model-{wandb_run.id}",
                model_path=best_model_path,
                metadata=dict(
                    epoch=best_epoch,
                    score=callback_checkpoint.best_model_score,
                    model_path=best_model_path,
                ),
                aliases=["best"],
            )

        print(
            f"Result: best_epoch={best_epoch} best_val_auroc={callback_checkpoint.best_model_score:.4f}"
        )

        return best_student

    @torch.no_grad()
    def post_training_sanitycheck(
        self,
        student: nn.Module,
        trainer: LayerwiseKDModelWrapper,
        checkpoint_callback: ModelCheckpoint,
        device: str,
    ):

        assert not student.training

        student.to(device)

        actual, _ = self.metric_func(student, self.dl_val, device=device, verbose=True)

        assert checkpoint_callback.best_model_score is not None
        expected = float(checkpoint_callback.best_model_score)

        np.testing.assert_allclose(
            [actual, np.max(trainer.arr_metrics["val_auroc"])],
            expected,
            err_msg="stats computed from modified student should match the last one returned from distillator",
        )

    @torch.no_grad()
    def log_test_metrics(
        self, best_student: nn.Module, logger: WandbLogger, device: str
    ):
        test_auroc, test_loss = self.metric_func(
            best_student,
            dataloader=self.dl_test,
            device=device,
            verbose=True,
        )
        logger.experiment.summary["student_test_auroc"] = test_auroc
        logger.experiment.summary["student_test_loss"] = test_loss

        return test_auroc, test_loss

    def log_model(
        self,
        wandb_run: Run,
        artifact_name: str,
        metadata: typing.Dict,
        model_path: str,
        aliases: typing.List[str],
    ):

        artifact = Artifact(artifact_name, type="model", metadata=metadata)
        artifact.add_file(model_path, name="model.ckpt")

        wandb_run.log_artifact(
            artifact,
            aliases=aliases,
        )
