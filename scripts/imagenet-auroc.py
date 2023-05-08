import os
import typing
import numpy as np
import click

from torch import nn
from tqdm import tqdm

from pathlib import Path

from datetime import datetime

import torchmetrics

from xaikd import datasets, utils
from xaikd.utils import click_types

import torch
from torchmetrics.classification import BinaryAUROC

from torch.utils.data import DataLoader, Subset, Dataset

from torchvision.datasets import ImageNet
from torchvision.models.resnet import ResNet18_Weights


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    num_classes: int,
    num_workers=4,
    batch_size=256,
) -> float:
    dl = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size)

    acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    for x, y in tqdm(dl):
        x = x.to(device)

        logits = model(x).detach().cpu()

        acc.update(logits, y)

    return float(acc.compute())


def compute_auroc(
    model: nn.Module, dataset: Dataset, class_pair: typing.Tuple[int, int], device: str
) -> typing.Tuple[float, float, int]:
    val_loader = DataLoader(dataset, batch_size=128, num_workers=2)

    click.echo(f"We have {len(val_loader)} items in the dataloader for AUROC!")

    c1, c2 = class_pair

    auroc = BinaryAUROC()
    with torch.no_grad():
        count = 0
        for x, y in tqdm(val_loader):
            logits = model(x.to(device)).detach().cpu()

            logodd = logits[:, c1] - logits[:, c2]
            count += y.shape[0]
            binary_targets = torch.where(y == c1, 0, 1)

            auroc.update(logodd, binary_targets)

    auroc = auroc.compute()

    return float(np.max([auroc, 1 - auroc])), float(auroc), count


@click.command()
@click.option("--model", type=click_types.Model(), required=True)
@click.option("--classes", type=str, required=True)
@click.option("--output-dir", default=Path("./tmp"), type=click_types.Path())
@click.option("--skip-accuracy", is_flag=True, default=False)
def main(model: nn.Module, classes, output_dir: Path, skip_accuracy: bool):
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    model = model.to(device)

    c1, c2 = np.array(classes.split("vs")).astype(int)

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    ds = ImageNet(root="./datasets/imagenet", split="val", transform=transform)

    if not skip_accuracy:
        accuracy = compute_accuracy(model, ds, device=device, num_classes=1000)
        click.echo(f"Accuracy(full dataset): {accuracy:.4f}")

    selected: typing.List[int] = (
        np.argwhere(np.isin(ds.targets, [c1, c2])).reshape(-1).tolist()
    )
    print(f"We have {len(selected)} selected images")

    auroc_corrected, auroc, count = compute_auroc(
        model=model,
        dataset=Subset(ds, indices=selected),
        device=device,
        class_pair=(c1, c2),
    )

    click.echo(
        f"ImageNet({classes}): auroc={auroc_corrected:.4f} (before corrected: {auroc:.4f})"
    )

    output_dir = output_dir / getattr(model, "__name")
    os.makedirs(output_dir, exist_ok=True)

    click.echo(f"Output: {output_dir}")

    utils.dump_json_with_string_serializer(
        output_dir / f"auroc-{classes}.json",
        dict(model=model, count=count, auroc_corrreted=auroc_corrected, auroc=auroc),
    )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
