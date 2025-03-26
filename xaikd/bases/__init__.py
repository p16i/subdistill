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

    for base_variant_cls in [
        PCA,
        PRCAPosDef,
        PRCAPosDefWeightSTDWithH0_5,
        PRCAPosDefWeightSTDWithH0_6,
        PRCAPosDefWeightSTDWithH0_7,
        PRCAPosDefWeightSTDWithH0_8,
        PRCAPosDefWeightSTDWithH0_9,
        PRCAPosDefWeightSTDWithH0_95,
        PRCAPosDefWeightSTDWithH1,
        GradPCAWeightSTDWithEntropy,
    ]:
        base_variant_slug = base_variant_cls.slug()
        slug = f"{base_variant_slug}--centered"

        assert not (slug in BASES)
        BASES[slug] = partial(base_variant_cls, centering=True)


_add_centering_variants()
