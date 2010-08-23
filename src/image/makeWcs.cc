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
 
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"

namespace except = lsst::pex::exceptions; 
namespace afwImg = lsst::afw::image;

using namespace std;

typedef lsst::daf::base::PropertySet PropertySet;
typedef lsst::afw::image::Wcs Wcs;
typedef lsst::afw::image::PointD PointD;


/// Use this function to create a Wcs object using a fits header. It examines the header and determines the 
/// most suitable object to return, either a general Wcs object, or a more specific object specialised to a 
/// given coordinate system (e.g TanWcs)
/// 
afwImg::Wcs::Ptr afwImg::makeWcs(PropertySet::Ptr fitsMetadata){
    std::string ctype1;
    if( fitsMetadata->exists("CTYPE1")) {
        ctype1 = fitsMetadata->getAsString("CTYPE1");
    } else {
        return afwImg::Wcs::Ptr(new afwImg::Wcs());
    }
    
    if( ctype1.substr(5, 3) == "TAN") {
        return afwImg::Wcs::Ptr(new afwImg::TanWcs(fitsMetadata));
    }
    
    //Default Wcs class
    return afwImg::Wcs::Ptr(new afwImg::Wcs(fitsMetadata));
}
    
