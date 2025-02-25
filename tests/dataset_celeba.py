import pytest


from xaikd import datasets


@pytest.mark.slow
@pytest.mark.parametrize("is_train", [True, False])
def test_callable_and_create_subsets(is_train):

    dataset = datasets.construct("celeba")
    assert True

    dataset.create_subset(train_split=is_train)
    assert True
