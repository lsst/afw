from __future__ import absolute_import

from ._coord import *

def __repr__(self):
    className = self.getClassName()
    coordSystem = self.getCoordSystem()
    argList = ["%r*afwGeom.degrees" % (pos.asDegrees(),) for pos in self]
    if coordSystem == TOPOCENTRIC:
#        topoCoord = TopocentricCoord.cast(self)
        argList += [
            repr(self.getEpoch()),
            "(%r)" % (self.getObservatory(),),
        ]
    elif coordSystem not in (ICRS, GALACTIC):
        argList.append(repr(self.getEpoch()))
    return "%s(%s)" % (self.getClassName(), ", ".join(argList))

Coord.__repr__ = __repr__
Fk5Coord.__repr__ = __repr__
IcrsCoord.__repr__ = __repr__
GalacticCoord.__repr__ = __repr__
EclipticCoord.__repr__ = __repr__
TopocentricCoord.__repr__ = __repr__

del __repr__

# Add __iter__ to allow  'ra,dec = coord' statement in python
def __iter__(self):
    for i in (0, 1):
        yield self[i]

Coord.__iter__ = __iter__
Fk5Coord.__iter__ = __iter__
IcrsCoord.__iter__ = __iter__
GalacticCoord.__iter__ = __iter__
EclipticCoord.__iter__ = __iter__
TopocentricCoord.__iter__ = __iter__

del __iter__

def __len__(self):
    return 2

Coord.__len__ = __len__
Fk5Coord.__len__ = __len__
IcrsCoord.__len__ = __len__
GalacticCoord.__len__ = __len__
EclipticCoord.__len__ = __len__
TopocentricCoord.__len__ = __len__

del __len__

# Add __reduce__ for Coord subclasses that take 3 arguments
def __reduce3__(self):
    return (TYPE, (self.getLongitude(), self.getLatitude(), self.getEpoch()))

Coord.__reduce__ = __reduce3__
Fk5Coord.__reduce__ = __reduce3__
EclipticCoord.__reduce__ = __reduce3__

del __reduce3__

# Add __reduce__ for Coord subclasses that take 2 arguments
def __reduce2__(self):
    return (TYPE, (self.getLongitude(), self.getLatitude()))

IcrsCoord.__reduce__ = __reduce2__
GalacticCoord.__reduce__ = __reduce2__

del __reduce2__
