import os
import typing
from functools import partial
import numpy as np
import numpy.typing as npt


from .orthogonal import *
from .orthogonal_weighting import *
from .register import get_basis, BASES
from .helpers import learn_basis
from .adapter import Adapter, AdapterMode
