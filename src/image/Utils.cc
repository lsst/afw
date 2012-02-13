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

#include "lsst/afw/fits.h"
#include "lsst/afw/image/Utils.h"

namespace lsst {
namespace afw {
namespace image {

/**
 * \brief Return the metadata from a fits file
 */
PTR(lsst::daf::base::PropertyList) readMetadata(
    std::string const& fileName, ///< File to read
    int hdu,               ///< HDU to read
    bool strip       ///< Should I strip e.g. NAXIS1 from header?
) {
    PTR(lsst::daf::base::PropertyList) metadata = boost::make_shared<lsst::daf::base::PropertyList>();
    afw::fits::Fits fitsfile(fileName, "r", afw::fits::Fits::AUTO_CHECK | afw::fits::Fits::AUTO_CLOSE);
    if (hdu == 0) {
        int naxis = 0;
        fitsfile.readKey("NAXIS", naxis);
        if (naxis == 0) {
            fitsfile.setHdu(2); // skip the first HDU because it's empty
        }
    } else {
        fitsfile.setHdu(hdu);
    }
    fitsfile.readMetadata(*metadata, strip);
    return metadata;
}
    
}}} // namespace lsst::afw::image
