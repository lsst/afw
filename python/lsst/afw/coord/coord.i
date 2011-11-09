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
#include "lsst/afw/image.h"    
#include "lsst/afw/geom.h"
#include "lsst/daf/base.h"    
#include "lsst/afw/coord/Utils.h"
#include "lsst/afw/coord/Coord.h"
%}


// The shared pointer declarations must precede the #inlude statement for Coord.h
%shared_ptr(lsst::afw::coord::Coord);
%shared_ptr(lsst::afw::coord::Fk5Coord);
%shared_ptr(lsst::afw::coord::IcrsCoord);
%shared_ptr(lsst::afw::coord::GalacticCoord);
%shared_ptr(lsst::afw::coord::EclipticCoord);
%shared_ptr(lsst::afw::coord::TopocentricCoord);


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

