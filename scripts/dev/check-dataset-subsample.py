import os
import click
from datetime import datetime

import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader

from xaikd import datasets, utils, attributors, models


@click.command()
@click.option("--dataset-name", default="cifar100-people", type=str)
@click.option("--training-size", default=0.1, type=float)
@click.option("--seed", default=1, type=int)
@click.option("--output-dir", default="./tmp", type=str)
def main(dataset_name, training_size, seed, output_dir):
    arguments = locals()
    start_time = datetime.now()

    click.echo(f"Subsample dataset {dataset_name} with training-size={training_size}")

    dataset = datasets.construct(dataset_name)

    ds_train = datasets.subsample_dataset(
        dataset.create_subset(train_split=True), ratio=training_size, seed=seed
    )

    output_dir = Path(output_dir) / f"{dataset}-ts{training_size}-seed{seed}"
    os.makedirs(output_dir, exist_ok=True)

    utils.dump_json(output_dir / "indices.json", dict(indices=ds_train.indices))

    teacher_model = "cifar100-resnet18-p1"
    layer = "layer3"

    train_loader = DataLoader(ds_train, batch_size=64)
    logit_mod = attributors.TargetClassEvidence(dataset=dataset)

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=models.get_trained_model(teacher_model),
        layer=layer,
        data_loader=train_loader,
        dataset=dataset,
        logit_modifier=logit_mod,
        device="cpu",
        rng=np.random.default_rng(seed=seed),
    )

    np.save(output_dir / "arr_act", arr_act)
    np.save(output_dir / "arr_ctx", arr_ctx)

    print(f"Artifact saved to {output_dir}")

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
