import typing

from dataclasses import dataclass

from xaikd import constants
from .interface import LayerPolicy, LastLayerPolicy

from .register import get_policy, get_last_layer_policy, policy_exists
from .previous_works import *
from .ours import *
from .ours_pca_linear import *


@dataclass
class LambdaCollection:
    lambda_task: float
    lambda_kd: float
    lambda_layer: float


def parse_layer_string(
    txt: typing.Optional[str],
) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """_summary_

    Args:
        txt (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        teacher_layers : typing.List[str]
        student_layers : typing.List[str]
    """
    if txt is None:
        return [], []

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


def resolve_lambdas_and_layer_policy(
    teacher: str,
    policy_name: str,
    lambda_layer: typing.Optional[float],
    default_lambda_layer_config: typing.Optional[str],
    layerwise_training: bool,
) -> typing.Tuple[LambdaCollection, str]:
    if policy_name == "student-only":
        lambda_collection = LambdaCollection(lambda_task=1, lambda_kd=0, lambda_layer=0)
        layer_policy = "nothing"
    elif policy_name == "kd-only":
        lambda_collection = LambdaCollection(lambda_task=0, lambda_kd=1, lambda_layer=0)
        layer_policy = "nothing"
    else:
        layer_policy = policy_name

        if layerwise_training:
            print(f"[layerwise_training={layerwise_training}]: we force lambda_layer=1")
            lambda_layer = 1
            lambda_kd = 1
        else:
            lambda_kd = 1
            if "wo-kd" in layer_policy:
                print(
                    f"[layerwise_training={layerwise_training}]: we force lambda_kd=0"
                )
                lambda_kd = 0

            layer_policy = layer_policy.replace("wo-kd", "")

            lambda_layer = constants.resolve_lambda_layer(
                teacher_model_name=teacher,
                policy_name=layer_policy,
                lambda_layer=lambda_layer,
                default_config_key=default_lambda_layer_config,
            )

        lambda_collection = LambdaCollection(
            lambda_task=0, lambda_kd=lambda_kd, lambda_layer=lambda_layer
        )

    assert policy_exists(layer_policy)

    return lambda_collection, layer_policy
