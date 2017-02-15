from __future__ import absolute_import

from ._coord import *

def __repr__(self):
    className = self.getClassName()
    coordSystem = self.getCoordSystem()
    argList = ["%r*afwGeom.degrees" % (pos.asDegrees(),) for pos in self]
    if coordSystem == TOPOCENTRIC:
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

def _reduceCoord(self):
    return (Coord, (self.getLongitude(), self.getLatitude(), self.getEpoch()))
Coord.__reduce__ = _reduceCoord

def _reduceFk5Coord(self):
    return (Fk5Coord, (self.getLongitude(), self.getLatitude(), self.getEpoch()))
Fk5Coord.__reduce__ = _reduceFk5Coord

def _reduceEclipticCoord(self):
    return (EclipticCoord, (self.getLongitude(), self.getLatitude(), self.getEpoch()))
EclipticCoord.__reduce__ = _reduceEclipticCoord

def _reduceGalacticCoord(self):
    return (GalacticCoord, (self.getLongitude(), self.getLatitude()))
GalacticCoord.__reduce__ = _reduceGalacticCoord

def _reduceIcrsCoord(self):
    return (IcrsCoord, (self.getLongitude(), self.getLatitude()))
IcrsCoord.__reduce__ = _reduceIcrsCoord

