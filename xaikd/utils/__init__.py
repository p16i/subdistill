import json
import torch

from . import interceptor


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def dump_json(dest: str, data: dict):
    with open(dest, "w") as fh:
        json.dump(
            data,
            fh,
            indent=4,
            sort_keys=True,
        )
