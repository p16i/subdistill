import click

from tqdm import tqdm
import torchmetrics

from datetime import datetime

import torch
from torch import nn

from xaikd import utils, bases, models

from xaikd import constants
from xaikd.constants import datasets


@torch.no_grad()
def compute_acc(
    model: nn.Module, dataset: datasets.DatasetConfiguration, device: str
) -> float:
    data_loader = dataset.loader(train_split=False)

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=dataset.num_classes)

    for x, y in data_loader:
        logits = model(x.to(device))

        metric.update(logits, y)

    metric = metric.compute()

    return float(metric.cpu().detach().numpy())


def attach_projected_fh(model: nn.Module, layer: str, basis_name, k):
    module = getattr(model, layer)[-1]

    U, mu = get_basis(layer, basis_name, k)

    UUT = torch.from_numpy(U @ U.T).float().to("cuda")

    UUT = UUT.unsqueeze(2).unsqueeze(3)

    mu = torch.from_numpy(mu).float().to("cuda").reshape((1, -1, 1, 1))

    def fh(mod, input, output):
        assert isinstance(output, torch.Tensor)

        projected = F.conv2d(output - mu, UUT)

        return projected + mu

    hook = module.register_forward_hook(fh)

    return hook


@click.command()
@click.option("--model", type=str)
@click.option("--layer", type=str)
@click.option("--artifact-dir", type=str)
@click.option("--basis-names", default=",".join(constants.BASIS_NAMES))
def main(model, layer, basis_names, artifact_dir):
    arguments = locals()

    start_time = datetime.now()

    device = utils.get_device()

    model_obj = models.get_model(model).to(device)

    dataset_name = model.split("-")[0]

    dataset = datasets.get_constant(dataset_name)

    click.echo(f"Load artifacts from `{artifact_dir}`")

    assert layer == "layer1"
    # how to get this number?
    dims = 64

    # todo: this has to be part of arch
    module: nn.Module = getattr(model_obj, layer)[-1]

    original_accuracy = compute_acc(model_obj, dataset, device)

    for basis_name in tqdm(
        basis_names.split(","), desc=f"[model={model},device={device}]"
    ):
        basis = bases.get_basis(basis_name)

        basis.load(artifact_dir, device=device)

        arr_accs = []
        for k in range(dims):
            try:
                hook = module.register_forward_hook(
                    basis.construct_fh_rank_k_projection(k)
                )
                acc = compute_acc(model_obj, dataset, device)
                arr_accs.append(acc)
            finally:
                hook.remove()

        utils.dump_json(
            f"{artifact_dir}/{basis}/accuracy.json",
            dict(accuracies=acc, dims=dims, original_accuracy=original_accuracy),
        )

    time_took = datetime.now() - start_time
    click.echo(f"Time Took: {time_took.seconds / 60:2.2f} minutes")


if __name__ == "__main__":
    main()
