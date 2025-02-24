import pytest
import numpy as np
import torch

from xaikd import attributors, datasets, logit_modifiers


def test_multiclass_target_logit():
    dataset = datasets.construct("cifar100-people")

    all_classes = set(range(dataset.num_classes))
    class1 = 1

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))

    actual = logit_modifiers.MultiClassTargetLogit()(logits, torch.tensor([class1] * 2))

    np.testing.assert_allclose(actual[:, class1], logits[:, class1])

    np.testing.assert_allclose(actual[:, list(all_classes.difference([class1]))], 0)


def test_multiclass_all_logits():
    torch.manual_seed(1)
    num_classes = 3

    expected = logits = torch.rand((10, num_classes))

    actual = logit_modifiers.MultiClassAllLogits()(logits, None)

    np.testing.assert_allclose(actual, expected)


def test_multiclass_winning_logit():
    logits = torch.tensor(
        [
            [0.3, 0.5, 0.9],
            [0.6, 0.3, 0.1],
            [0.6, 0.8, 0.2],
        ]
    )

    logits_mod_single = logit_modifiers.MultiClassWinningLogit()(logits, None)

    np.testing.assert_allclose(
        logits_mod_single,
        [
            [0, 0.0, 0.9],
            [0.6, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ],
    )


def test_multiclass_zero_logit():
    logits = torch.tensor(
        [
            [0.3, 0.5, 0.9],
            [0.6, 0.3, 0.1],
            [0.6, 0.8, 0.2],
        ]
    )

    actual = logit_modifiers.MultiClassZeroLogit()(logits, None)

    np.testing.assert_allclose(actual, torch.zeros_like(logits))


def test_multiclass_differencetop2logits():
    logits = torch.tensor(
        [
            [0.3, 0.5, 0.9],
            [0.6, 0.3, 0.1],
            [0.6, 0.8, 0.2],
        ]
    )

    actual = logit_modifiers.MultiClassDifferenceTop2Logits()(logits, None)

    np.testing.assert_allclose(
        actual,
        [
            [0, -0.5, 0.9],
            [0.6, -0.3, 0.0],
            [-0.6, 0.8, 0.0],
        ],
    )


def test_multiclass_logoddwinning():
    logits = torch.tensor(
        [
            [3, 5, 9],
            [6, 3, 1],
            [6, 8, 2],
        ]
    ).float()

    actual = logit_modifiers.MultiClassLogOddWinning()(logits, None)

    expected_logit_winning = torch.tensor([9, 6, 8.0])
    expected_logit_others = torch.tensor(
        [
            [3, 5],
            [3, 1],
            [6, 2],
        ]
    )

    expected = expected_logit_winning - torch.logsumexp(expected_logit_others, dim=1)

    np.testing.assert_allclose(actual, expected)


# todo: binary logodd winning
