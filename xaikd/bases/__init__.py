import os
import typing
from functools import partial
import numpy as np
import numpy.typing as npt


from .orthogonal import *
from .orthogonal_weighting import *
from .register import get_basis, BASES
from .helpers import learn_basis


def _add_centering_variants():

    # fixme: add all variants of with entropy
    for base_variant_cls in [PCA, PRCAPosDef, PRCAPosDefWeightSTDWithH0_95]:
        base_variant_slug = base_variant_cls.slug()
        slug = f"{base_variant_slug}--centered"

        assert not (slug in BASES)
        BASES[slug] = partial(base_variant_cls, centering=True)


_add_centering_variants()
