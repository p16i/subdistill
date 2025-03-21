import typing

from .interface import LayerPolicy, LastLayerPolicy

from .register import get_policy, get_last_layer_policy
from .previous_works import *
from .ours import *


# todo: move this to utils
def parse_layer_string(txt: str) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """_summary_

    Args:
        txt (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        teacher_layers : typing.List[str]
        student_layers : typing.List[str]
    """
    teacher_layers = []
    student_layers = []

    for layer in txt.split(","):
        slugs = layer.split(":")

        if len(slugs) == 1:
            teacher_layer = student_layer = slugs[0]
        elif len(slugs) == 2:
            teacher_layer, student_layer = slugs
        else:
            raise ValueError(
                "Could not parse `{layer}` into teacher and student layers!"
            )

        teacher_layers.append(teacher_layer)
        student_layers.append(student_layer)

    assert len(teacher_layers) == len(student_layers)

    return teacher_layers, student_layers
