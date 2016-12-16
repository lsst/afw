from __future__ import absolute_import, division, print_function

import numpy as np

from ._base import BaseCatalog
from ._match import SimpleMatch, ReferenceMatch, SourceMatch
from ._schema import Schema


def __repr__(self):
    return "Match(%s,\n            %s,\n            %g)" % \
        (repr(self.first), repr(self.second), self.distance)


def __str__(self):
    def sourceRaDec(s):
        if hasattr(s, "getRa") and hasattr(s, "getDec"):
            return " RA,Dec=(%g,%g) deg" % (s.getRa().asDegrees(), s.getDec().asDegrees())
        return ""

    def sourceXy(s):
        if hasattr(s, "getX") and hasattr(s, "getY"):
            return " x,y=(%g,%g)" % (s.getX(), s.getY())
        return ""

    def sourceStr(s):
        return s.__class__.__name__ + ("(id %d" % s.getId()) + sourceRaDec(s) + sourceXy(s) + ")"

    return "Match(%s, %s, dist %g)" % (sourceStr(self.first), sourceStr(self.second), self.distance,)


def __getitem__(self, i):
    """Treat a Match as a tuple of length 3: (first, second, distance)"""
    if i > 2 or i < -3:
        raise IndexError(i)
    if i < 0:
        i += 3
    if i == 0:
        return self.first
    elif i == 1:
        return self.second
    else:
        return self.distance


def __setitem__(self, i, val):
    """Treat a Match as a tuple of length 3: (first, second, distance)"""
    if i > 2 or i < -3:
        raise IndexError(i)
    if i < 0:
        i += 3
    if i == 0:
        self.first = val
    elif i == 1:
        self.second = val
    else:
        self.distance = val


def __len__(self):
    return 3


def clone(self):
    return self.__class__(self.first, self.second, self.distance)


def __getstate__(self):
    return self.first, self.second, self.distance


def __setstate__(self, state):
    self.__init__(*state)


for matchCls in (SimpleMatch, ReferenceMatch, SourceMatch):
    matchCls.__repr__ = __repr__
    matchCls.__str__ = __str__
    matchCls.__getitem__ = __getitem__
    matchCls.__setitem__ = __setitem__
    matchCls.__len__ = __len__
    matchCls.clone = clone
    matchCls.__getstate__ = __getstate__
    matchCls.__setstate__ = __setstate__


def packMatches(matches):
    """Make a catalog of matches from a sequence of matches.

    The catalog contains three fields:
    - first: the ID of the first source record in each match
    - second: the ID of the second source record in each match
    - distance: the distance of each match

    @param[in] matches  Sequence of matches, typically of type SimpleMatch, ReferenceMatch or SourceMatch.
        Each element must support: `.first.getId()`->int, `.second.getId()->int` and `.distance->float`.
    @return a catalog of matches.

    @note this pure python implementation exists because SWIG could not easily be used to wrap
    the overloaded C++ functions, so this was written and tested. It might be practical
    to wrap the overloaded C++ functions with pybind11, but there didn't seem much point.
    """
    schema = Schema()
    outKey1 = schema.addField("first", type=np.int64, doc="ID for first source record in match.")
    outKey2 = schema.addField("second", type=np.int64, doc="ID for second source record in match.")
    keyD = schema.addField("distance", type=np.float64, doc="Distance between matches sources.")
    result = BaseCatalog(schema)
    result.table.preallocate(len(matches))
    for match in matches:
        record = result.addNew()
        record.set(outKey1, match.first.getId())
        record.set(outKey2, match.second.getId())
        record.set(keyD, match.distance)
    return result
