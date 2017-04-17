from __future__ import absolute_import, division, print_function

from ._background import Background

__all__ = []  # import this module only for its side effects


def __reduce__(self):
    """Pickling"""
    return self.__class__, (self.getImageBBox(), self.getStatsImage())

Background.__reduce__ = __reduce__
