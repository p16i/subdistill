from .interface import Policy, LastLayerPolicy

LAYER_POLICY = dict()


def register_policy(name):
    """Decorator to register a layer policy"""

    def wrapped(fn):
        """Wrapped function to register a layer policy provider with`name`"""
        assert name not in LAYER_POLICY

        LAYER_POLICY[name] = fn

        return fn

    return wrapped


def get_policy(name: str, device: str, **kwargs) -> Policy:
    return LAYER_POLICY[name](device=device, **kwargs)
