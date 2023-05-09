import os
import pytest

from xaikd import datasets


@pytest.mark.parametrize("name", ["cifar10", "cifar100", "imagenet"])
@pytest.mark.skipif(
    not int(os.getenv("ASSERT_DATADIR", "0")),
    reason="run when ASSERT_DATADIR=1",
)
def test_construct_basedataset(name):
    datasets.construct(name).create_dataset(train_split=False)
    datasets.construct(name).create_dataset(train_split=True)

    # if we reach here, everything should be ok.
    # remark: perhaps, it is better to assert that no exceptive raised.
    assert True
