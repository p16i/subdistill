import os
import numpy as np
import click
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from copy import deepcopy

from torchvision import transforms

from pathlib import Path

import pandas as pd

from xaikd import (
    models,
    datasets,
    attributors,
    distillators,
    augmentations,
    approximators,
    distillation_info,
    utils,
    bases,
)
from xaikd.approximators import ApproximatorMode
from xaikd.showcases import cleverhans
from xaikd.distillation_info import ExperimentConfiguration


BASIS_MODE = "centered"


@click.command()
@click.option("--output-dir", type=Path, default="./tmp/showcase-cleverhans")
@click.option("--epochs", type=int, default=100)
@click.option("--contamination-level", type=float, default=0.75)
@click.option("--alphas", type=str, default="0.0,0.25,0.5,0.75,1.0")
@click.option("--training-size", type=float, default=0.1)
@click.option("--seed", type=int, default=1)
@click.option("--lr", type=float, default=0.0005)
@click.option("--basis-names", default="pca,prca-recon,pcaprca-recon,random")
@click.option("--compression-ratio", type=float, default=8)
def main(
    output_dir: Path,
    epochs,
    contamination_level,
    alphas,
    basis_names,
    seed,
    training_size,
    lr,
    compression_ratio,
):
    arguments = locals()
    start_time = datetime.now()

    arr_alphas = np.array(alphas.split(",")).astype(float)

    model_name = "cifar100-resnet18-p1"
    dataset_name = "cifar100-people"
    layer = "layer3"

    output_dir = (
        output_dir
        / f"{dataset_name}-ts{training_size}-clv{contamination_level}-seed{seed}"
        / model_name
        / layer
    )

    basis_names = basis_names.split(",")
    device = utils.get_device()

    teacher_model = models.get_model(model_name)
    teacher_model.to(device)
    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(dataset_name)

    logit_mod = attributors.OneClassEvidence(dataset)

    clean_train_ds = datasets.subsample_dataset(
        dataset.create_subset(train_split=True),
        ratio=training_size,
        seed=seed,
    )

    val_loader = datasets.build_dataloader(
        dataset.create_subset(train_split=False), shuffle=False
    )

    augmentation = augmentations.get_augmentation_for(dataset=dataset)

    contaminated_train_ds = cleverhans.contaminate_dataset(
        dataset=clean_train_ds, contamination_level=contamination_level, seed=seed
    )

    # todo: this can be abstract away
    # perhaps, make it a method of `dataset``
    contaminated_train_ds_with_aug = deepcopy(contaminated_train_ds)
    contaminated_train_ds_with_aug.dataset.transforms = transforms.Compose(
        [
            *augmentation,
            contaminated_train_ds_with_aug,
        ]
    )

    train_loader = datasets.build_dataloader(contaminated_train_ds, shuffle=True)
    train_loader_with_aug = datasets.build_dataloader(
        contaminated_train_ds_with_aug,
        shuffle=True,
        batch_size=int(np.ceil(64 * training_size)),
    )

    arr_act, arr_ctx = attributors.extract_activation_context(
        model=teacher_model,
        layer=layer,
        data_loader=train_loader,
        dataset=dataset,
        logit_modifier=logit_mod,
        device=device,
        rng=np.random.default_rng(seed=seed),
    )
    mean = np.mean(arr_act, axis=0)
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_dir / "act_mean", mean)

    distill_info = distillation_info.get_distill_infor(
        arch=model_name, layer=layer, compression_ratio=compression_ratio
    )

    arr_experiment_confs = [
        ExperimentConfiguration(
            basis_name="identity--centered",
            compression_ratio=1.0,
            approximator_mode=ApproximatorMode.HOMOGENOUS,
        ),
        ExperimentConfiguration(
            basis_name="identity--centered",
            compression_ratio=compression_ratio,
            approximator_mode=ApproximatorMode.HOMOGENOUS_LOWRANK_ADAPTER,
        ),
    ]

    for basis_name in basis_names:
        basis_name = f"{basis_name}--{BASIS_MODE}"
        arr_experiment_confs.append(
            ExperimentConfiguration(
                basis_name=basis_name,
                compression_ratio=compression_ratio,
                approximator_mode=ApproximatorMode.HOMOGENOUS_LOWRANK,
            ),
        )

    for basis_name in np.unique(
        list(map(lambda n: n.basis_name, arr_experiment_confs))
    ):
        print(f"Learning {basis_name}")
        basis = bases.get_basis(basis_name, seed=seed)

        basis.fit(arr_act, arr_ctx, mean=mean, device=device)
        basis.save(output_dir)

    for alpha in arr_alphas:
        lambda_mse = alpha
        lambda_xent = 1 - alpha

        ref_acc = None

        for conf in arr_experiment_confs:
            approximator = approximators.construct_approximator_for(
                teacher_model,
                layer=layer,
                compression_ratio=conf.compression_ratio,
                mode=conf.approximator_mode,
                seed=seed,
            )

            distillator = distillators.Layerwise(
                teacher=models.get_model(model_name),
                dataset=dataset,
                train_dataloader=train_loader_with_aug,
                val_dataloader=val_loader,
                device=device,
                weight_decay=0.0,
            )

            if ref_acc is None:
                ref_acc = distillator.ref_acc
            else:
                assert (
                    distillator.ref_acc == ref_acc
                ), "Reference models have different accuracy!"

            basis_name = conf.basis_name
            approximator_mode = approximators.normalize_mode_name(
                conf.approximator_mode
            )

            basis_distillation_output_dir = (
                output_dir
                / "distillation"
                / f"{approximator_mode}-comp{conf.compression_ratio}-ldmse{lambda_mse}-ldxent{lambda_xent}"
                / basis_name
            )

            os.makedirs(basis_distillation_output_dir, exist_ok=True)

            basis = bases.get_basis(basis_name, seed=seed)

            basis.load(output_dir)

            student = models.get_model(model_name)

            log_dir = basis_distillation_output_dir / "log"

            student, results = distillator.distill(
                student=student,
                approximator=approximator,
                distill_info=distill_info,
                epochs=epochs,
                basis=basis,
                device=device,
                lr=lr,
                logger=TensorBoardLogger(log_dir),
                log_dir=log_dir,
                lambda_mse=lambda_mse,
                lambda_xent=lambda_xent,
            )

            last_epoch_val_acc = results["arr_metrics"]["val"][-1]

            print(
                f"Result (contamination level={contamination_level}, lambda_mse={lambda_mse}, lambda_xent={lambda_xent}): Student with `{approximator_mode}` and `{basis}` acc={last_epoch_val_acc:.4f}"
            )

            arr_targets, arr_preds = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    arr_targets.append(y.numpy())
                    logits = student(x.to(device))
                    ypred = torch.argmax(logits, dim=1).cpu().numpy()
                    arr_preds.append(ypred)

            arr_targets = np.concatenate(arr_targets)
            arr_preds = np.concatenate(arr_preds)

            utils.dump_json(basis_distillation_output_dir / "results.json", results)

            pd.DataFrame.from_dict(dict(target=arr_targets, pred=arr_preds)).to_csv(
                basis_distillation_output_dir / "predictions.csv", index=False
            )

    print(f"Artifact save at: {output_dir}")
    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
