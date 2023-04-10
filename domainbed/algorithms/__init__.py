from .algorithms import *
from .miro import MIRO
from .distill_clip import (
    DFC_STAGE1,
    DFC_STAGE2,
    DFC_STAGE2_MIRO,
    DFC_STAGE31,
    DFC_STAGE32,
    DFC_STAGE33,
    DFC_STAGE34,
    DFC_STAGE35,
    DFC_STAGE3_MIRO,
)
from .distill_clip import CLIP
from .baselines import *


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
