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
 
#include "boost/make_shared.hpp"
// #include "Eigen/Core.h"
#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/image/TanWcs.h"

namespace except = lsst::pex::exceptions; 
namespace afwImg = lsst::afw::image;

using namespace std;

/**
 * Create a Wcs object from a fits header.
 * It examines the header and determines the 
 * most suitable object to return, either a general Wcs object, or a more specific object specialised to a 
 * given coordinate system (e.g TanWcs)
 */
afwImg::Wcs::Ptr afwImg::makeWcs(
        lsst::daf::base::PropertySet::Ptr metadata, ///< input metadata
        bool stripMetadata                              ///< Remove FITS keywords from metadata?
                                )
{
    std::string ctype1;
    if (metadata->exists("CTYPE1")) {
        ctype1 = metadata->getAsString("CTYPE1");
    } else {
        return boost::make_shared<afwImg::Wcs>();
    }

    afwImg::Wcs::Ptr wcs;
    if( ctype1.substr(5, 3) == "TAN") {
        wcs = afwImg::Wcs::Ptr(new afwImg::TanWcs(metadata)); // can't use make_shared as ctor is private
    } else {
        wcs = afwImg::Wcs::Ptr(new afwImg::Wcs(metadata));
    }

    if (stripMetadata) {
        afwImg::detail::stripWcsKeywords(metadata, wcs);
    }

    return wcs;
}
    
/**
 * Create a Wcs object from crval, crpix, CD, using CD elements (useful from python)
 *
 */
afwImg::Wcs::Ptr afwImg::makeWcs(
                                 lsst::afw::geom::PointD crval,
                                 lsst::afw::geom::PointD crpix,
                                 double CD11, double CD12, double CD21, double CD22
                                ) {
    Eigen::Matrix2d CD;
    CD << CD11, CD12, CD21, CD22;
    return afwImg::Wcs::Ptr(new lsst::afw::image::Wcs(crval, crpix, CD));
}
