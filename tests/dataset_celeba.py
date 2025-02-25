import pytest


from xaikd import datasets


@pytest.mark.slow
@pytest.mark.parametrize("is_train", [True, False])
def test_celeba_callable_and_create_subsets(is_train):

    dataset = datasets.construct("celeba")
    assert True

    dataset.create_subset(train_split=is_train)
    assert True


@pytest.mark.slow
@pytest.mark.parametrize("attr_ix", [0, 20, 39])
@pytest.mark.parametrize("is_train", [True, False])
def test_celeba_attr_callable_and_create_subsets(attr_ix, is_train):
    batch_size = 28

    dataset = datasets.construct(f"celeba-attr{attr_ix}")

    dataset.create_subset(train_split=is_train)

    ds = dataset.create_subset(train_split=is_train)
    _, y = next(
        iter(datasets.build_dataloader(ds, batch_size=batch_size, shuffle=False))
    )

    assert y.shape == (batch_size,), y.shape
