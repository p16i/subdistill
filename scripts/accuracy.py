import typing
import click
import os


from collections import OrderedDict
from pathlib import Path

from datetime import datetime


import numpy as np
import pandas as pd

from xaikd import (
    utils,
    models,
    datasets,
    metrics,
)
from xaikd.utils import click_types


@click.command()
@click.option("--arch", default="cifar100-resnet18-v1", type=str)
@click.option("--dataset-name", required=True, type=str)
def main(
    arch: str,
    dataset_name: str,
):
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    dataset = datasets.construct(dataset_name)

    model = models.get_trained_model(arch)

    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)

    model.eval()
    model.to(device)

    dl_val = datasets.build_dataloader(
        dataset.create_subset(train_split=False),
        shuffle=False,
    )

    metric = metrics.MetricAccuracy(num_classes=len(dataset.selected_classes))

    (ref_acc,) = metric(model=model, dataloader=dl_val, device=device, verbose=True)
    print(f"{arch} on {dataset_name}: ref_acc={ref_acc:.4f}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
