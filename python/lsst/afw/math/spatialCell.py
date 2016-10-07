from __future__ import absolute_import
from lsst.pex.exceptions import NotFoundError
#import lsst.afw.image as afwImage
from ._spatialCell import *

def spatialCellCandidateIter(self):
    while True:
        try:
            yield self.__deref__()
        except NotFoundError:
            raise StopIteration
        self.__incr__()

def spatialCellGetitem(self, idx):
    l = [x for x in self.begin()]
    return [x for x in self.begin()][idx]

def spatialCellIter(self):
    return self.begin().__iter__()

SpatialCellCandidateIterator.__iter__ = spatialCellCandidateIter
SpatialCell.__getitem__ = spatialCellGetitem
SpatialCell.__iter__ = spatialCellIter

def getCandidateRating(self):
    return self._flux

def setCandidateRating(self, flux):
    self._flux = flux

SpatialCellCandidate.getCandidateRating = getCandidateRating
SpatialCellCandidate.setCandidateRating = setCandidateRating