from tqdm import tqdm
import numpy as np
import torch

from torchvision.datasets import MNIST
from torchvision import transforms as T
from torch import nn
from torch.utils.data import DataLoader, Subset

from xaikd import datasets, attributors, utils

from zennit.composites import EpsilonGammaBox
from zennit.attribution import Gradient

from . import approximator


SEED = 1
NUMBER_CLASSES = 10
NUM_WORKERS = 2
BATCH_SIZE = 64

CONSIDERED_CLASSES = [4, 9]
CONSIDERED_LAYER = "act1"
NUM_SPATIAL_LOCATIONS_SAMPLING = 10


DATASET_DIR = "./datasets"

MODEL_URLS = {
    "mnist-k14-h128": "https://tubcloud.tu-berlin.de/s/84X83BZ7STJy28D/download/cnn-mnist-colab-ks14.pth"
}

DATA_MEAN, DATA_STD = 0.5, 0.5

MAIN_TRANSFORM = T.Compose([T.ToTensor(), T.Normalize((DATA_MEAN,), (DATA_STD,))])
INPUT_LOW_VALUE, INPUT_HIGH_VALUE = MAIN_TRANSFORM.transforms[1](
    torch.tensor([[[[[0.0]]] * 1], [[[[1.0]]] * 1]])
)


ARRAY_KS = np.arange(1, 10 + 1)


BASIS_CONSIDERED = ["pca", "prca-abs", "prca-recon"]


LOGIT_MODIFIER = attributors.LogOddEvidence(tuple(CONSIDERED_CLASSES))


def get_model(model_name):
    slugs = model_name.split("-")

    kernel_size = int(slugs[1][1:])
    nhidden = int(slugs[2][1:])

    model = CNN(kernel_size=kernel_size, nhidden=nhidden)

    model.load_state_dict(torch.hub.load_state_dict_from_url(MODEL_URLS[model_name]))

    model.eval()

    return model


class CNN(nn.Module):
    def __init__(self, input_channels=1, kernel_size=7, nhidden=128):
        super().__init__()

        self.kernel_size = kernel_size

        if kernel_size == 7:
            nsteps = 22
        elif kernel_size == 14:
            nsteps = 15
        elif kernel_size == 28:
            nsteps = 1

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=nhidden,
            kernel_size=kernel_size,
            padding="valid",
        )
        self.act1 = nn.ReLU()

        self.lin2 = nn.Linear((15**2) * (nhidden), 10)

    def forward_feat(self, x):
        x = self.act1(self.conv1(x))

        return x

    def forward(self, x):
        x = self.forward_feat(x)

        x = x.flatten(start_dim=1)
        x = self.lin2(x)

        return x

    def __str__(self):
        oc, _, kw, kh = self.conv1.weight.shape

        return f"CNN(ks={(kw,kh)},nhidden={oc})"


def get_loaders():
    train_ds = MNIST(
        root="./datasets", train=True, transform=MAIN_TRANSFORM, download=True
    )

    val_ds = MNIST(
        root="./datasets", train=False, transform=MAIN_TRANSFORM, download=True
    )

    return (
        DataLoader(
            train_ds, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True
        ),
        DataLoader(
            val_ds, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=False
        ),
    )


def build_subclasses_loader(considered_classes, samples_per_class):
    np.random.seed(SEED)
    train_ds = MNIST(
        root="./datasets", train=True, transform=MAIN_TRANSFORM, download=True
    )

    train_subset = Subset(
        train_ds,
        indices=datasets.selected_subset_samples_for_classes(
            train_ds.targets,
            considered_classes,
            samples_per_class=samples_per_class,
        ).tolist(),
    )

    val_ds = MNIST(
        root="./datasets", train=False, transform=MAIN_TRANSFORM, download=True
    )

    val_subset = Subset(
        val_ds,
        indices=np.argwhere(np.isin(val_ds.targets.numpy(), considered_classes))
        .reshape(-1)
        .tolist(),
    )

    return train_subset, val_subset


def extract_activaiton_and_context(
    model: nn.Module, layer: str, train_subset, device: str
):
    arr_act = []
    arr_ctx = []

    module = getattr(model, layer)

    composite = EpsilonGammaBox(low=INPUT_LOW_VALUE, high=INPUT_HIGH_VALUE)

    for x, y in tqdm(
        DataLoader(train_subset, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
    ):
        try:
            module, hook = utils.interceptor.attach_hook_intercept_module(module)

            x = x.to(device)

            with Gradient(model=model, composite=composite) as attributor:
                _ = attributor.forward(x, lambda logits: LOGIT_MODIFIER(logits))

            act = utils.interceptor.get_output(module)
            rel = act.grad

            ctx = torch.where(act.abs() > 0, rel / act, 0)

            act, ctx = utils.subsample_tensors(
                act.detach().cpu().numpy(),
                ctx.detach().cpu().numpy(),
                num_locations=NUM_SPATIAL_LOCATIONS_SAMPLING,
            )

            arr_act.append(act)
            arr_ctx.append(ctx)

        finally:
            hook.remove()

    arr_act = np.vstack(arr_act)

    arr_ctx = np.vstack(arr_ctx)

    return arr_act, arr_ctx
