from collections import OrderedDict
from functools import partial

import torchvision
import torch

import wandb

from . import MODEL_GENERATORS


CELEBA_NUM_ATTRIBUTES = 40
WANDB_PROJECT = "xaikd-training-teacher-models"


def get_state_dict(run_id: str) -> OrderedDict:
    agent = wandb.Api()

    # todo: host this somewhere on tubclound or others
    artifact: wandb.Artifact = agent.artifact(f"{WANDB_PROJECT}/model-{run_id}:latest")

    artifact_dir = artifact.download(root="/tmp")

    ckpt = torch.load(
        f"{artifact_dir}/model.ckpt",
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    state_dict = ckpt["state_dict"]

    new_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("encoder.", "")
        new_dict[new_k] = v

    return new_dict


def _get_resnet18_celeba_run_id(run_id):
    state_dict = get_state_dict(run_id)

    model = torchvision.models.resnet18(weights=None, num_classes=CELEBA_NUM_ATTRIBUTES)

    model.load_state_dict(state_dict)

    setattr(model, "num_classes", CELEBA_NUM_ATTRIBUTES)

    model.eval()

    return model


def ano():
    for slug, run_id in [
        ("celeba-resnet18-scratch", "n8r0q2vb"),
        ("celeba-resnet18-pretrained", "6oj5aaxl"),
    ]:
        MODEL_GENERATORS[slug] = partial(_get_resnet18_celeba_run_id, run_id=run_id)


ano()
