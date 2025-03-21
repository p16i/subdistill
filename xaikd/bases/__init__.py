import os
import typing
from functools import partial
import numpy as np
import numpy.typing as npt


from .orthogonal import *
from .orthogonal_weighting import *
from .register import get_basis
from .helpers import learn_basis


# fixme: remove all this code below

# # from . import pcalookahead
# # from xaikd.bases import pcalookahead
# # from xaikd.bases.learners import (
# #     PRCAGreedyLearner,
# #     PRCAReconGreedy,
# #     PRCASignAlignGreedy,
# #     PRCASignAlignGreedyV2,
# # )

# def _add_centering_variants():

#     for base_variant_cls in [PCA]:
#         base_variant_slug = base_variant_cls.slug()
#         slug = f"{base_variant_slug}centering"

#         assert not (slug in BASES)
#         BASES[slug] = partial(base_variant_cls, centering=True)


# _add_centering_variants()
