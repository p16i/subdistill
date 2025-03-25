from . import DatasetConfiguration

DATASET_REGISTRY = dict()


def add_dataset_to_registry(name, cls):
    assert name not in DATASET_REGISTRY
    DATASET_REGISTRY[name] = cls


def register_dataset(name):
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        add_dataset_to_registry(name, cls)
        # this return is necessary for class inheritance
        return cls

    return wrapped


def construct(name: str) -> DatasetConfiguration:
    assert name in DATASET_REGISTRY, f"dataset={name} does not exist!"

    dataset = DATASET_REGISTRY[name]()
    setattr(dataset, "__name", name)

    return dataset
