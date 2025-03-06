import typing

import pandas as pd

from xaikd import constants


DF_SUPERCLASS_MAPPING = pd.read_csv(constants.CIFAR100_SUPER_CLASS_MAPPING)


def get_fineclass_names_indices_of_superclass(
    superclass: str,
) -> typing.Tuple[typing.List[str], typing.List[int]]:
    assert superclass in constants.CIFAR100_SUPER_CLASSES

    df_selected = DF_SUPERCLASS_MAPPING[
        DF_SUPERCLASS_MAPPING.coarse_label_name == superclass
    ]

    df_selected = df_selected.sort_values(by="fine_label")

    arr_idx = df_selected.fine_label.values.tolist()
    arr_names = df_selected.fine_label_name.values.tolist()

    return arr_names, arr_idx


from . import original, subclasses, some_vs_others
