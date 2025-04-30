import numpy as np
import pytest

from xaikd.distillators import trainer_wrapper


@pytest.mark.parametrize(
    "strategy,current_epoch,lambda_layer,expected",
    [
        ("n2n", 0, 10, (False, 10)),
        ("n2n", 20, 10, (False, 10)),
        ("n2n", 50, 10, (False, 10)),
        ("layerwise", 0, 10, (True, 1)),
        ("layerwise", 20, 10, (True, 1)),
        ("layerwise", 50, 10, (True, 1)),
        ("n2n@50", 20, 10, (True, 1)),
        ("n2n@50", 49, 10, (True, 1)),
        ("n2n@50", 50, 10, (False, 0)),
        ("n2n@50", 100, 10, (False, 0)),
        ("n2n@20", 19, 10, (True, 1)),
        ("n2n@20", 20, 10, (False, 0)),
    ],
)
def test_should_detach_output(strategy, current_epoch, lambda_layer, expected):
    expected_detach, expected_lambda_layer = expected

    actual_detach, actual_lambda_layer = (
        trainer_wrapper.resolve_detach_layer_output_and_lambda_layer(
            strategy, current_epoch=current_epoch, lambda_layer=lambda_layer
        )
    )
    np.testing.assert_equal(actual_detach, expected_detach)
    np.testing.assert_equal(actual_lambda_layer, expected_lambda_layer)
