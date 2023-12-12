""" Layer/Module Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
from itertools import repeat
#fixed https://github.com/rwightman/pytorch-image-models/commit/94ca140b67cb602ee7e146af32bfb63b60df96f4#diff-c7abf83bc43184f6101237b08d7c489c361f3d57b3538d633f6f01d35254b73c
import collections.abc


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

