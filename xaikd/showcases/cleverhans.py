import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import CIFAR100

from PIL import Image, ImageDraw, ImageFont

from copy import deepcopy


CLEVER_HAN_SYMBOL = "+"
COLOR = "red"
LOCATION = (20, 20)


def add_cleverhan_symbol(img):
    copied_img = img.copy()

    ImageDraw.Draw(copied_img).text(LOCATION, CLEVER_HAN_SYMBOL, COLOR, fontsize=15)

    return copied_img


def contaminate_dataset(
    dataset: Subset, contamination_level: float, seed: int
) -> Subset:
    """_summary_

    Args:
        dataset (Subset): _description_
        contamination_level (float): _description_
        seed (int): _description_

    Returns:
        Subset: _description_
    """
    assert isinstance(dataset, Subset) and isinstance(dataset.dataset, CIFAR100)
    assert 0 <= contamination_level <= 1.0

    rng = np.random.default_rng(seed=seed)

    subsampled_indices = dataset.indices

    contaminated_dataset = deepcopy(dataset.dataset)

    data = contaminated_dataset.data
    print("data.shape", data.shape)
    targets = contaminated_dataset.targets
    if isinstance(targets, list):
        targets = np.array(targets)

    # we use the convention that the class whose label index is smallest is the victim target.
    victim_class_idx = np.min(targets)

    print(
        f"Victim Class for Contamination(level={contamination_level}): {victim_class_idx}"
    )

    all_possible_victim_sample_indices = np.argwhere(
        targets == victim_class_idx
    ).reshape(-1)

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
        targets[subsampled_indices] == victim_class_idx
    ).sum()

    assert (
        contamination_level - 0.1
        <= (sample_indices_with_symbol.shape[0] / num_samples_belong_victim_class)
        <= contamination_level
    )

    for ix in sample_indices_with_symbol:
        img = Image.fromarray(data[ix])

        new_img = add_cleverhan_symbol(img)

        data[ix] = np.array(new_img)

    return Subset(dataset=contaminated_dataset, indices=dataset.indices)
