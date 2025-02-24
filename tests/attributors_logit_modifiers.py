import pytest
import numpy as np
import torch

from xaikd import attributors, datasets


def test_logit_modifier_oneclass():
    dataset = datasets.construct("cifar100-people")

    all_classes = set(range(dataset.num_classes))
    class1 = 1

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))

    logits_mod_single = attributors.TargetClassEvidence(
        num_classes=dataset.num_classes
    )(logits, torch.tensor([class1] * 2))

    assert (logits_mod_single[:, class1] == logits[:, class1]).all()
    assert (logits_mod_single[:, list(all_classes.difference([class1]))] == 0).all()


@pytest.mark.skip(reason="obsolete")
@pytest.mark.parametrize("target", ("abc", None))
def test_logit_modifier_logodd(target):
    dataset: datasets.TwoClassesDataset = datasets.construct("cifar100-1vs8")

    all_classes = set(range(dataset.num_classes))
    class1, class2 = dataset.selected_classes

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))

    logits_mod_logood = attributors.LogOddEvidence((class1, class2))(logits, target)

    assert (logits_mod_logood[:, class1] == logits[:, class1]).all()
    assert (logits_mod_logood[:, class2] == -logits[:, class2]).all()
    assert (
        logits_mod_logood[:, list(all_classes.difference([class1, class2]))] == 0
    ).all()


def test_logit_modifier_selected_classes():
    dataset: datasets.Cifar100SuperClassesDataset = datasets.construct(
        "cifar100-people"
    )

    # selected classes in the new label index system.
    selected_classes = np.array([0, 1])

    assert (selected_classes <= (dataset.num_classes - 1)).all()

    all_classes = set(range(dataset.num_classes))

    torch.manual_seed(1)

    logits = torch.rand((2, dataset.num_classes))
    logits[:, selected_classes] += 1

    logits_mod_single = attributors.SelectedClassesEvidence(
        selected_classes=selected_classes.tolist()
    )(logits, None)

    assert (logits_mod_single[:, selected_classes] > 0).all()
    assert (
        logits_mod_single[:, list(all_classes.difference(selected_classes))] == 0
    ).all()


def test_logit_modifier_winningclass():
    num_classes = 3

    logits = torch.tensor(
        [
            [0.3, 0.5, 0.9],
            [0.6, 0.3, 0.1],
            [0.6, 0.8, 0.2],
        ]
    )

    logits_mod_single = attributors.WinningClassEvidence(num_classes=num_classes)(
        logits, None
    )

    np.testing.assert_allclose(
        logits_mod_single,
        [
            [0, 0.0, 0.9],
            [0.6, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ],
    )


def test_logit_modifier_differencetop2winningclasses():
    num_classes = 3

    logits = torch.tensor(
        [
            [0.3, 0.5, 0.9],
            [0.6, 0.3, 0.1],
            [0.6, 0.8, 0.2],
        ]
    )

    logits_mod_single = attributors.DifferenceTop2WinningClassesEvidence(
        num_classes=num_classes
    )(logits, None)

    np.testing.assert_allclose(
        logits_mod_single,
        [
            [0, -0.5, 0.9],
            [0.6, -0.3, 0.0],
            [-0.6, 0.8, 0.0],
        ],
    )


def test_logit_modifier_multi_logoddwinning():
    logits = torch.tensor(
        [
            [3, 5, 9],
            [6, 3, 1],
            [6, 8, 2],
        ]
    ).float()

    actual = attributors.MultiClassLogOddWinning()(logits, None)

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
