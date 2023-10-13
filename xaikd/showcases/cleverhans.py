import typing


import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

from PIL import Image, ImageDraw

from copy import deepcopy


CLEVER_HAN_SYMBOL = "+"
COLOR = "red"


def add_cleverhan_symbol(img, rng: np.random.Generator):
    copied_img = img.copy()

    x = rng.integers(low=0, high=31 - 4)
    # remark: because the anchor attribute doesn't seem to work with the default font,
    # we therefoe adjust by - 3 manually here to compensate the empty space above `+` from the default font.
    # cf. ./notebooks/2023-10-s16/dev-add-symbol-to-img.ipynb
    y = rng.integers(low=0 - 3, high=31 - 4 - 3)

    location = (x, y)

    ImageDraw.Draw(copied_img).text(
        location,
        text=CLEVER_HAN_SYMBOL,
        fill=COLOR,
    )
    return copied_img


def contaminate_dataset(
    dataset: Subset,
    contamination_level: float,
    seed: int,
    victim_class_indices: typing.List[int],
) -> Subset:
    """_summary_

    Args:
        dataset (Subset): _description_
        contamination_level (float): _description_
        seed (int): _description_

    Returns:
        Subset: _description_
    """
    assert isinstance(dataset, Subset)

    if contamination_level > 0:
        isinstance(dataset.dataset, CIFAR100)

    assert 0 <= contamination_level <= 1.0

    rng = np.random.default_rng(seed=seed)

    # indices of samples belong to selected for the Subset
    subsampled_indices = dataset.indices

    contaminated_dataset = deepcopy(dataset.dataset)

    data = contaminated_dataset.data
    print("data.shape", data.shape)
    targets = contaminated_dataset.targets
    if isinstance(targets, list):
        targets = np.array(targets)

    print(
        f"Victim Classes for Contamination(level={contamination_level}): {victim_class_indices}"
    )

    all_possible_victim_sample_indices = np.argwhere(
        np.isin(targets, victim_class_indices)
    ).reshape(-1)

    # only consider samples that belong to the subset.
    potential_victim_sample_indices = np.array(
        list(set(all_possible_victim_sample_indices).intersection(subsampled_indices))
    )

    total_victim_samples = np.floor(
        potential_victim_sample_indices.shape[0] * contamination_level
    ).astype(int)

    sample_indices_with_symbol = rng.permutation(potential_victim_sample_indices)[
        :total_victim_samples
    ]

    num_samples_belong_victim_class = (
        np.isin(targets[subsampled_indices], victim_class_indices)
    ).sum()

    assert (
        contamination_level - 0.1
        <= (sample_indices_with_symbol.shape[0] / num_samples_belong_victim_class)
        <= contamination_level
    )

    print(f"> {len(sample_indices_with_symbol)} victims (total={total_victim_samples})")

    for ix in sample_indices_with_symbol:
        img = Image.fromarray(data[ix])

        new_img = add_cleverhan_symbol(img, rng)

        data[ix] = np.array(new_img)

    return Subset(dataset=contaminated_dataset, indices=dataset.indices)
