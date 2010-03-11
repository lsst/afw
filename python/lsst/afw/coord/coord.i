// -*- lsst-c++ -*-

%{
#include "lsst/afw/image.h"    
#include "lsst/afw/geom.h"
#include "lsst/afw/coord/Utils.h"
#include "lsst/afw/coord/Coord.h"
%}


// The shared pointer declarations must precede the #inlude statement for Coord.h
SWIG_SHARED_PTR(Coord, lsst::afw::coord::Coord);
%define %declareDerived(COORDTYPE)
SWIG_SHARED_PTR_DERIVED(COORDTYPE##Coord, lsst::afw::coord::Coord, lsst::afw::coord::COORDTYPE##Coord);
%enddef

%declareDerived(Fk5);
%declareDerived(Icrs);
%declareDerived(Galactic);
%declareDerived(Ecliptic);
%declareDerived(AltAz);

%include "lsst/afw/coord/Utils.h"
%include "lsst/afw/coord/Coord.h"

