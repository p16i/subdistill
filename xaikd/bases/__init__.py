import os
import typing
from functools import partial
import numpy as np
import numpy.typing as npt


from .orthogonal import *
from .orthogonal_weighting import *
from .pcalookahead import PCALookAhead
from .register import get_basis, BASES
from .helpers import learn_basis
from .adapter import Adapter, AdapterMode


def resolve_basis_name_for_layer(slug: str, layer: str) -> str:
    arr_layer_conf = slug.split(",")
    if len(arr_layer_conf) == 1:
        return slug
    else:
        dict_layer_basis = dict()
        for conf in arr_layer_conf:
            _layer, _name = conf.split("@")
            dict_layer_basis[_layer] = _name
        return dict_layer_basis[layer]
