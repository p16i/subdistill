from xaikd.constants import datasets


def test():
    const_dataset = datasets.get_constant("cifar10")

    assert hasattr(const_dataset, "num_classes")
    assert hasattr(const_dataset, "transformation")
