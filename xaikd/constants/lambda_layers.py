import typing


def resolve_lambda_layer(
    teacher_model_name: str,
    policy_name: str,
    lambda_layer: typing.Union[float, None],
    default_config_key: typing.Union[str, None],
) -> float:

    if lambda_layer is not None:
        return lambda_layer
    else:
        assert (
            default_config_key is not None
        ), "default_config should be specified when lambda_layer is none."

        lambda_layer = DEFAULT_LAMBDA_LAYER[default_config_key][teacher_model_name][
            policy_name
        ]
        print(
            f"Resolve `lambda_layer` from config:{default_config_key}[{teacher_model_name}][{policy_name}]"
        )
        assert lambda_layer is not None

        return lambda_layer


DEFAULT_LAMBDA_LAYER = {
    "dummy": {
        "cifar100-resnet18-v1": {
            "policy-1": 0.1,
            "policy-2": 0.7,
            "vid": 0.8,
        }
    },
    "cifar100-some-vs-others--clean": {
        "cifar100-resnet18-v1": {
            "nothing": 0,
            "vid": 1.0,
            "basis-bn:pca": 1.0,
            "basis-bn:gradpca": 0.001,
            "basis-bn:prcaposdef": 1.0,
        }
    },
    "cifar100-some-vs-others--small": {
        "cifar100-resnet18-v1": {
            "nothing": 0,
            "vid": 10.0,
            "basis-bn:pca": 10.0,
            "basis-bn:gradpca": 0.1,
            "basis-bn:prcaposdef": 1.0,
        }
    },
}
