// -*- lsst-c++ -*-

/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 *
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program.  If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */


%{
#include "lsst/base.h" // for PTR
#include "lsst/afw/image.h"
#include "lsst/afw/geom.h"
#include "lsst/daf/base.h"
#include "lsst/afw/coord/Coord.h"
%}

%include "std_vector.i"

%useValueEquality(lsst::afw::coord::Coord);

// The shared pointer declarations must precede the %include statement for Coord.h
%shared_ptr(lsst::afw::coord::Coord);
%shared_ptr(lsst::afw::coord::Fk5Coord);
%shared_ptr(lsst::afw::coord::IcrsCoord);
%shared_ptr(lsst::afw::coord::GalacticCoord);
%shared_ptr(lsst::afw::coord::EclipticCoord);
%shared_ptr(lsst::afw::coord::TopocentricCoord);

%template(CoordVector) std::vector<PTR(lsst::afw::coord::Coord const)>;

%rename(__getitem__) lsst::afw::coord::Coord::operator[];

// -----------------------------------------------------------------------
// THESE CASTS ARE NOW DEPRECATED; USE E.G. `Fk5Coord.cast()` INSTEAD
%inline %{
    PTR(lsst::afw::coord::Fk5Coord) cast_Fk5(PTR(lsst::afw::coord::Coord) c) {
        return std::dynamic_pointer_cast<lsst::afw::coord::Fk5Coord>(c);
    }
    PTR(lsst::afw::coord::IcrsCoord) cast_Icrs(PTR(lsst::afw::coord::Coord) c) {
        return std::dynamic_pointer_cast<lsst::afw::coord::IcrsCoord>(c);
    }
    PTR(lsst::afw::coord::GalacticCoord) cast_Galactic(PTR(lsst::afw::coord::Coord) c) {
        return std::dynamic_pointer_cast<lsst::afw::coord::GalacticCoord>(c);
    }
    PTR(lsst::afw::coord::EclipticCoord) cast_Ecliptic(PTR(lsst::afw::coord::Coord) c) {
        return std::dynamic_pointer_cast<lsst::afw::coord::EclipticCoord>(c);
    }
%}
// -----------------------------------------------------------------------

%castShared(lsst::afw::coord::Fk5Coord, lsst::afw::coord::Coord)
%castShared(lsst::afw::coord::IcrsCoord, lsst::afw::coord::Coord)
%castShared(lsst::afw::coord::GalacticCoord, lsst::afw::coord::Coord)
%castShared(lsst::afw::coord::EclipticCoord, lsst::afw::coord::Coord)
%castShared(lsst::afw::coord::TopocentricCoord, lsst::afw::coord::Coord)

%include "lsst/afw/coord/Coord.h"

// add __str__ and __repr__ methods to a specified Coord class (e.g. Fk5Coord)
%define strCoord(TYPE)
%extend lsst::afw::coord::TYPE {
    std::string __str__() const {
        std::ostringstream os;
        os << (*self);
        return os.str();
    }
    %pythoncode %{
def __repr__(self):
    className = self.getClassName()
    coordSystem = self.getCoordSystem()
    argList = ["%r*afwGeom.degrees" % (pos.asDegrees(),) for pos in self]
    if coordSystem == TOPOCENTRIC:
        topoCoord = TopocentricCoord.cast(self)
        argList += [
            repr(self.getEpoch()),
            "(%r)" % (topoCoord.getObservatory(),),
        ]
    elif coordSystem not in (ICRS, GALACTIC):
        argList.append(repr(self.getEpoch()))
    return "%s(%s)" % (self.getClassName(), ", ".join(argList))
    %}
}
%enddef

strCoord(Coord);
strCoord(Fk5Coord);
strCoord(IcrsCoord);
strCoord(GalacticCoord);
strCoord(EclipticCoord);
strCoord(TopocentricCoord);

// Add __iter__ to allow  'ra,dec = coord' statement in python
%define genCoord(TYPE)
%extend lsst::afw::coord::TYPE {
    %pythoncode %{
def __iter__(self):
    for i in (0, 1):
        yield self[i]
def __len__(self):
    return 2
    %}
}
%enddef

genCoord(Coord);
genCoord(Fk5Coord);
genCoord(IcrsCoord);
genCoord(GalacticCoord);
genCoord(EclipticCoord);
genCoord(TopocentricCoord);


// Add __reduce__ for Coord subclasses that take 3 arguments
%define reduceCoord3(TYPE)
%extend lsst::afw::coord::TYPE {
    %pythoncode %{
def __reduce__(self):
    return (TYPE, (self.getLongitude(), self.getLatitude(), self.getEpoch()))
    %}
}
%enddef

// Add __reduce__ for Coord subclasses that take 2 arguments
%define reduceCoord2(TYPE)
%extend lsst::afw::coord::TYPE {
    %pythoncode %{
def __reduce__(self):
    return (TYPE, (self.getLongitude(), self.getLatitude()))
    %}
}
%enddef

reduceCoord3(Coord);
reduceCoord3(Fk5Coord);
reduceCoord2(IcrsCoord);
reduceCoord2(GalacticCoord);
reduceCoord3(EclipticCoord);
