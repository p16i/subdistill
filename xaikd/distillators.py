import copy
import os
import typing

from dataclasses import dataclass
import pytorch_lightning as pl
import numpy as np


import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from xaikd import utils, datasets, attributors, bases
from xaikd.utils import metrics


LAYERS = ["layer1", "layer2", "layer3", "layer4"]
EPOCHS_PER_LAYER = 10


@dataclass
class LayerDistillInfo:
    layer_name: str
    num_input_channels: int
    num_output_channels: int
    output_spatial_dims: typing.Tuple[int, int]


class ApproximationModule(nn.Module):
    def __init__(
        self,
        adapter: typing.Callable,
        num_input_channels: int,
        num_output_channels: int,
        output_spatial_dims: typing.Tuple[int, int],
        kernel_size=3,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            num_input_channels,
            num_output_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.act1 = nn.ReLU()
        self.pool1 = nn.AdaptiveAvgPool2d(output_spatial_dims)
        self.conv2 = nn.Conv2d(
            num_output_channels, num_output_channels, kernel_size=1, padding="valid"
        )

        # conv1x1
        self.adapter = adapter

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)

        x = self.adapter(x)

        return x

    def remove_adapter(self):
        self.adapter = nn.Identity()


class Grafting:
    def __init__(
        self,
        teacher: torch.nn.Module,
        dataset: datasets.TwoClassesDataset,
        basis_dir: str,
        compression_rate: float,
        device: str,
    ) -> None:
        pass
        self.teacher = teacher
        self.dataset = dataset

        self.compression_rate = compression_rate

        self.device = device

    def distill(
        self, epochs: int, basis_name: str, basis_dir: Path, device: str, seed=1
    ):
        utils.deactivate_requires_grad(self.teacher)

        # todo: deep copy should not change any
        student = copy.deepcopy(self.teacher)
        student.to(device)

        ref_auroc = metrics.estimate_auroc(
            student,
            self.dataset.loader(train_split=False),
            attributors.LogOddEvidence(self.dataset.selected_classes, self.dataset),
            self.device,
        )

        arr_distill_info = self.setup()

        global_epoch_ix = 0

        arr_metrics = []

        epochs_per_layer = epochs // len(LAYERS)
        print(
            f"[Grafting] with {epochs_per_layer} epochs per layer (total {len(LAYERS)} layers)"
        )
        (
            total_teacher_params,
            total_teacher_trainable_params,
        ) = utils.count_params_in_model(self.teacher)

        assert total_teacher_trainable_params == 0

        torch.manual_seed(seed)
        for distill_info in tqdm(arr_distill_info):
            approxer = self.on_training_layer_start(
                student, distill_info, basis_name, basis_dir, device
            )

            count_total_params, count_trainable_params = utils.count_params_in_model(
                student
            )

            print(distill_info)
            print(
                f"> total_params: {count_total_params} (trainable {count_trainable_params})"
            )

            assert (
                count_trainable_params > 0
                and count_trainable_params < total_teacher_params
            )

            # Optimizers specified in the torch.optim package
            optimizer = torch.optim.SGD(approxer.parameters(), lr=0.0001)

            tbar = tqdm(total=epochs_per_layer)
            for epoch in range(epochs_per_layer):
                for x, y in self.dataset.loader(train_split=True, shuffle=True):
                    logits = student(x.to(device))

                    loss = F.cross_entropy(logits, y.to(device))

                    loss.backward()

                    optimizer.step()

                    # backword
                global_epoch_ix += 1

                auroc = metrics.estimate_auroc(
                    student,
                    self.dataset.loader(train_split=False),
                    attributors.LogOddEvidence(
                        self.dataset.selected_classes, self.dataset
                    ),
                    self.device,
                )

                auroc = np.max([auroc, 1 - auroc])

                tbar.update(1)
                tbar.set_description(
                    f"[AUROC={auroc:.4f} (teacher: {ref_auroc:.4f})]"
                )

                arr_metrics.append(
                    dict(
                        layer=distill_info.layer_name,
                        global_epoch=global_epoch_ix,
                        layer_epoch=epoch,
                        auroc=auroc,
                        teacher_auroc=ref_auroc,
                    )
                )
            if distill_info != "layer4":
                approxer.remove_adapter()

        return arr_metrics

    def setup(self) -> typing.List[LayerDistillInfo]:
        # remark: hardcode for now
        return [
            LayerDistillInfo(
                layer_name="layer1",
                num_input_channels=64,
                num_output_channels=int(64 * self.compression_rate),
                output_spatial_dims=(32, 32),
            ),
            LayerDistillInfo(
                layer_name="layer2",
                num_input_channels=int(64 * self.compression_rate),
                num_output_channels=int(128 * self.compression_rate),
                output_spatial_dims=(16, 16),
            ),
            LayerDistillInfo(
                layer_name="layer3",
                num_input_channels=int(128 * self.compression_rate),
                num_output_channels=int(256 * self.compression_rate),
                output_spatial_dims=(8, 8),
            ),
            LayerDistillInfo(
                layer_name="layer4",
                num_input_channels=int(256 * self.compression_rate),
                num_output_channels=int(512 * self.compression_rate),
                output_spatial_dims=(4, 4),
            ),
        ]

    def on_training_layer_start(
        self,
        student: nn.Module,
        distil_info: LayerDistillInfo,
        basis_name: str,
        basis_dir: Path,
        device: str,
    ):
        utils.deactivate_requires_grad(student)

        basis = bases.get_basis(basis_name)

        basis.load(artifact_dir=basis_dir / distil_info.layer_name, device=device)

        approx_mod = ApproximationModule(
            adapter=basis.contruct_rank_d_decoder(distil_info.num_output_channels),
            num_input_channels=distil_info.num_input_channels,
            num_output_channels=distil_info.num_output_channels,
            output_spatial_dims=distil_info.output_spatial_dims,
        )

        approx_mod.to(device)

        setattr(student, distil_info.layer_name, approx_mod)

        return approx_mod

    def on_training_layer_end(self, approxer: ApproximationModule):
        approxer.remove_adapter()
