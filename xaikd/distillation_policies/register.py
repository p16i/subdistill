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


def policy_exists(name: str) -> bool:
    # for our policy, `name=basis-bn-sum-normalized:<basis_name>`
    slugs = name.split(":")

    actual_policy_name = slugs[0]

    return actual_policy_name in LAYER_POLICY


def get_last_layer_policy(name: str, **kwargs) -> LastLayerPolicy:
    return LAYER_POLICY[name](device=None, **kwargs)
