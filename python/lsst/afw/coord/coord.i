// -*- lsst-c++ -*-

%{
#include "lsst/afw/image.h"    
#include "lsst/afw/geom.h"
#include "lsst/daf/base.h"    
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
%declareDerived(Equatorial);
%declareDerived(Galactic);
%declareDerived(Ecliptic);
%declareDerived(AltAz);


%rename(__getitem__) lsst::afw::coord::Coord::operator[];

// make dynamic casts available for each Coord type.
// If the makeCoord factory returns a pointer to the base class,
// ... and access to the derived members is required, these can be used to cast.
%inline %{
    lsst::afw::coord::Fk5Coord::Ptr cast_Fk5(lsst::afw::coord::Coord::Ptr c) {
        return boost::shared_dynamic_cast<lsst::afw::coord::Fk5Coord>(c);
    }
    lsst::afw::coord::IcrsCoord::Ptr cast_Icrs(lsst::afw::coord::Coord::Ptr c) {
        return boost::shared_dynamic_cast<lsst::afw::coord::IcrsCoord>(c);
    }
    lsst::afw::coord::EquatorialCoord::Ptr cast_Equatorial(lsst::afw::coord::Coord::Ptr c) {
        return boost::shared_dynamic_cast<lsst::afw::coord::EquatorialCoord>(c);
    }
    lsst::afw::coord::GalacticCoord::Ptr cast_Galactic(lsst::afw::coord::Coord::Ptr c) {
        return boost::shared_dynamic_cast<lsst::afw::coord::GalacticCoord>(c);
    }
    lsst::afw::coord::EclipticCoord::Ptr cast_Ecliptic(lsst::afw::coord::Coord::Ptr c) {
        return boost::shared_dynamic_cast<lsst::afw::coord::EclipticCoord>(c);
    }
    // altaz omitted as it requires an Observatory and DateTime as well.
    // ie. it's not similar enough to use the same factory
%}

%include "lsst/afw/coord/Utils.h"
%include "lsst/afw/coord/Coord.h"

