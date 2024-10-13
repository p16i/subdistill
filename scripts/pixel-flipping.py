import os
import click
from pathlib import Path


from datetime import datetime

from copy import deepcopy
import torch
from torch.utils.data import random_split

from torchvision import transforms
from PIL import Image

from tqdm import tqdm
import pandas as pd

from xaikd import datasets, models, explainers, utils, attributors

SEED = 1


@click.command()
@click.option("--model-name", type=str)
@click.option("--explainer-name", type=str)
@click.option("--dataset-name", default="imagenet-butterfly")
@click.option("--data-percentage", default=0.01)
@click.option("--artifact-dir", default="/tmp")
def main(model_name, explainer_name, dataset_name, data_percentage, artifact_dir):
    arguments = locals()
    start_time = datetime.now()
    rng = torch.Generator()
    rng.manual_seed(1)

    device = utils.get_device()

    click.echo(f"Performing Pixelflipping for {model_name} with {explainer_name}")

    dataset = datasets.construct(dataset_name)

    output_dir = Path(artifact_dir) / dataset_name / model_name

    ds_train = dataset.create_subset(train_split=True)
    ds_train, _ = random_split(
        ds_train, [data_percentage, 1 - data_percentage], generator=rng
    )

    dl = datasets.build_dataloader(ds_train, shuffle=False)

    model = models.get_trained_model(model_name)
    utils.modify_last_layer_for_subclasses(model, dataset.selected_classes)
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    explainer = explainers.get_explainer(
        explainer_name,
        model=model,
        normalizer=transforms.Normalize(*dataset.input_statistics),
    )

    arr_logits, arr_heatmaps = explainer.explain(
        dataloader=dl,
        logit_modifier=attributors.TargetClassEvidence(
            num_classes=len(dataset.selected_classes)
        ),
        device=device,
    )

    n_data_points = arr_logits.shape[0]

    arr_dfs = []

    ds_train_only_crop = deepcopy(ds_train)
    ds_train_only_crop.dataset.transform = transforms.Compose(
        [
            transforms.Resize(ds_train.dataset.transform.resize_size),
            transforms.CenterCrop(ds_train.dataset.transform.crop_size),
        ]
    )

    baseline, transform_for_perturbed_image = (
        utils.pixelflipping.get_baseline_and_transform_for_perturbed_img(
            transforms.Normalize(*dataset.input_statistics)
        )
    )

    for dix in tqdm(range(n_data_points), desc="performing pixel flipping"):
        img, target = ds_train_only_crop[dix]
        assert isinstance(img, Image.Image), type(img)

        arr_num_pixels, arr_logits = utils.pixelflipping.perform_pixel_flipping(
            model=model,
            img=img,
            heatmap=arr_heatmaps[dix],
            target=target,
            baseline=baseline,
            transform=transform_for_perturbed_image,
            device=device,
        )

        n_steps = arr_num_pixels.shape[0]

        df = pd.DataFrame(
            dict(
                dix=[dix] * n_steps,
                target=[target] * n_steps,
                logit=arr_logits,
                num_flipped_pixels=arr_num_pixels,
            )
        )

        arr_dfs.append(df)

    df_final = pd.concat(arr_dfs)

    time_took = datetime.now() - start_time

    df_final.to_csv(output_dir / f"{explainer_name}.csv", index=False)

    click.echo(f"Result save to  {output_dir}")
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
