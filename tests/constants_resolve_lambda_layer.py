import pytest

from xaikd import constants


@pytest.mark.parametrize(
    "lambda_layer,expected,teacher_name,policy_name,config_key",
    [
        (1, 1, "cifar100-resnet18-v1", None, None),
        (
            None,
            constants.lambda_layers.DEFAULT_LAMBDA_LAYER["dummy"][
                "cifar100-resnet18-v1"
            ]["policy-1"],
            "cifar100-resnet18-v1",
            "policy-1",
            "dummy",
        ),
    ],
)
def test_resolve_lambda_layer(
    lambda_layer, expected, teacher_name, policy_name, config_key
):

    actual = constants.resolve_lambda_layer(
        lambda_layer=lambda_layer,
        teacher_model_name=teacher_name,
        policy_name=policy_name,
        default_config_key=config_key,
    )

    assert actual == expected


def test_resolve_lambda_layer_failed():
    lambda_layer = None
    policy_name = "novel"
    config_key = "dummy"
    teacher_name = "cifar100-resnet18-v1"

    with pytest.raises(KeyError):
        constants.resolve_lambda_layer(
            teacher_model_name=teacher_name,
            lambda_layer=lambda_layer,
            policy_name=policy_name,
            default_config_key=config_key,
        )
