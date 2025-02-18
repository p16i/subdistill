import os
import typing
from functools import partial
import numpy as np
import numpy.typing as npt


from .orthogonal import *
from .register import BASES

# from . import pcalookahead
# from xaikd.bases import pcalookahead
# from xaikd.bases.learners import (
#     PRCAGreedyLearner,
#     PRCAReconGreedy,
#     PRCASignAlignGreedy,
#     PRCASignAlignGreedyV2,
# )


def get_basis(basis_name, **kwargs) -> OrthogonalBasis:

    basis = BASES[basis_name](**kwargs)

    return basis


def _add_centering_variants():

    for base_variant_cls in [PCA]:
        base_variant_slug = base_variant_cls.slug()
        slug = f"{base_variant_slug}centering"

        assert not (slug in BASES)
        BASES[slug] = partial(base_variant_cls, centering=True)


_add_centering_variants()
