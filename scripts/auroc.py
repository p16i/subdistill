import numpy as np
import click

from torch import nn
from tqdm import tqdm

from pathlib import Path

from datetime import datetime

from xaikd import datasets, utils
from xaikd.utils import click_types

import torch
from torchmetrics.classification import BinaryAUROC

from torch.utils.data import DataLoader

from torchvision.datasets import ImageNet
from torchvision.models.resnet import ResNet18_Weights


@click.command()
@click.option("--model", type=click_types.Model(), required=True)
@click.option("--classes", default="322,323", type=str, required=True)
@click.option("--output-dir", default=Path("./tmp"), type=click_types.Path())
def main(model: nn.Module, classes, output_dir: Path):
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    model = model.to(device)

    c1, c2 = np.array(classes.split(",")).astype(int)

    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    ds = ImageNet(root="./datasets/imagenet", split="val", transform=transform)
    val_loader = DataLoader(ds, batch_size=128, num_workers=2)

    auroc = BinaryAUROC()
    with torch.no_grad():
        count = 0
        for x, y in tqdm(val_loader):
            logits = model(x.to(device)).detach().cpu()

            logodd = logits[:, c1] - logits[:, c2]
            count += y.shape[0]

            auroc.update(logodd, y)

    auroc = auroc.compute()

    auroc_corrreted = np.max([auroc, 1 - auroc])

    click.echo(f"We have {len(val_loader)} items in the dataloader!")

    # todo: write output

    click.echo(
        f"ImageNet({classes}): auroc={auroc_corrreted:.4f} (before corrected: {auroc:.4f})"
    )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
