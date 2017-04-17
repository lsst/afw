from __future__ import absolute_import, division, print_function

from lsst.pex.exceptions import NotFoundError
from ._spatialCell import SpatialCellCandidateIterator, SpatialCell

__all__ = []  # import this module only for its side effects


def spatialCellCandidateIter(self):
    while True:
        try:
            yield self.__deref__()
        except NotFoundError:
            return
        self.__incr__()
SpatialCellCandidateIterator.__iter__ = spatialCellCandidateIter


def spatialCellIter(self):
    return self.begin().__iter__()
SpatialCell.__iter__ = spatialCellIter


def spatialCellGetitem(self, idx):
    idx = int(idx)
    num_cells = len(self)
    if idx < -num_cells or idx >= num_cells:
        raise IndexError("idx={} < -{} or >= {})".format(idx, num_cells, num_cells))
    if idx < 0:
        idx += num_cells
    for i, cell in enumerate(self):
        if i >= idx:
            return cell
SpatialCell.__getitem__ = spatialCellGetitem
