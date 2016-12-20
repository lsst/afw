from __future__ import absolute_import

from ._background import Background

__all__ = []

def __reduce__(self):
    """Pickling"""
    return self.__class__, (self.getImageBBox(), self.getStatsImage())

Background.__reduce__ = __reduce__

