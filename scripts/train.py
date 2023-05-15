import click
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F

from xaikd import models, datasets, utils, attributors
from xaikd.utils import metrics

from tqdm import tqdm


class Lenet5(nn.Module):
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(16, 120, kernel_size=5, padding="valid"),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(120, 60),
            # nn.Tanh(),
        )

        self.lin2 = nn.Linear(60, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.lin2(x)

        return x


@click.command()
@click.option("--epochs", type=int, default=20)
@click.option("--lr", type=float, default=0.001)
def main(epochs, lr):
    arguments = locals()
    start_time = datetime.now()

    device = utils.get_device()

    dataset: datasets.TwoClassesDataset = datasets.construct("cifar100-35vs98")

    model = Lenet5(3, num_classes=100)
    model.to(device)

    tbar = tqdm(total=epochs)

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x, y in dataset.loader(train_split=True, shuffle=True):
            logits = model(x.to(device))

            loss = F.cross_entropy(logits, y.to(device))
            loss.backward()
            optimizer.step()

        auroc = metrics.estimate_auroc(
            model,
            dataset.loader(train_split=False),
            attributors.LogOddEvidence(dataset.selected_classes, dataset),
            device,
        )

        tbar.update(1)

        tbar.set_description(f"auroc={auroc:.4f}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
