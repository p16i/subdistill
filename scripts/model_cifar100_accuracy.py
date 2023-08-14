import click
from datetime import datetime

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, MeanMetric
from torch.utils.data import DataLoader

from xaikd import models, datasets
from xaikd import utils

import typing


def compute_xent_and_accuracy(
    model: nn.Module, dl: DataLoader, num_classes=100, device="cpu"
) -> typing.Tuple[float, float]:
    model.eval()
    metric = Accuracy(task="multiclass", num_classes=num_classes)
    metric_xent = MeanMetric()
    for x, y in dl:
        logits = model(x.to(device)).cpu()

        metric_xent.update(F.cross_entropy(logits, y))
        metric.update(logits, y)

    acc = float(metric.compute())
    xent = float(metric_xent.compute())

    return xent, acc


@click.command()
@click.option("--model-name", type=str)
def main(model_name):
    arguments = locals()
    start_time = datetime.now()

    click.echo("Hello, main!")

    device = utils.get_device()

    model = models.get_model(model_name)
    model.to(device=device)

    dataset = datasets.construct("cifar100")

    train_dl = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )

    val_dl = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )

    print(f"Model={model_name}")
    for prefix, dl in [("train", train_dl), ("val", val_dl)]:
        xent, acc = compute_xent_and_accuracy(model, dl, device=device)
        print(f" > [{prefix:5s}] xent={xent:.4f} acc={acc:.4f}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
