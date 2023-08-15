import os
from pathlib import Path
import click
from datetime import datetime

from xaikd import models, datasets
from xaikd.utils import metrics
import pandas as pd

from tqdm import tqdm
from xaikd import utils


@click.command()
@click.option("--dataset-name", type=str)
@click.option("--model-names", type=str)
@click.option("--output-dir", type=str, default="./tmp")
def main(dataset_name, model_names, output_dir):
    arguments = locals()
    start_time = datetime.now()

    model_names = model_names.split(",")

    click.echo(
        f"Computing accuracy on `{dataset_name}` for these `{len(model_names)}` models"
    )

    dataset = datasets.construct(dataset_name)

    train_dl = datasets.build_dataloader(
        dataset.create_subset(train_split=True), shuffle=False
    )

    val_dl = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    device = utils.get_device()

    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    stats = []

    for model_name in tqdm(model_names):
        model = models.get_model(model_name)
        model.to(device)

        print(f"Model={model_name}")
        model_stat = dict(model=model_name)
        for prefix, dl in [("train", train_dl), ("val", val_dl)]:
            xent, acc = compute_xent_and_accuracy(model, dataset, dl, device=device)
            acc, xent = metrics.accuracy_with_subclasses(
                model,
                dl,
                considered_classes=dataset.selected_classes,
                transform_target=dataset.transform_target,
                device=device,
            )
            print(f" > [{prefix:5s}] xent={xent:.4f} acc={acc:.4f}")

            model_stat[f"{prefix}_xent"] = xent
            model_stat[f"{prefix}_acc"] = acc

        stats.append(model_stat)

    output_path = output_dir / f"model-accuracy-{dataset_name}.csv"
    df = pd.DataFrame(stats)
    df.to_csv(output_path, index=False)

    click.echo(f"Check output at {output_path}")
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
