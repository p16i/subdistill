BASES = dict()


def register_basis():
    """Decorator to register a data modality provider."""

    def wrapped(cls):
        """Wrapped function to register a data modality provider with name `name`"""

        slug = cls.slug()

        assert not (slug in BASES)

        BASES[slug] = cls

        return cls

    return wrapped
