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

/**
 * \file
 * \brief A set of classes of general utility in connection with images
 *
 * We provide representations of points, bounding boxes, circles etc.
 */
#ifndef LSST_AFW_IMAGE_UTILS_H
#define LSST_AFW_IMAGE_UTILS_H

#include <list>
#include <map>
#include <string>
#include <utility>

#include "boost/format.hpp"
#include "boost/mpl/bool.hpp"
#include <memory>

#include "lsst/afw/image/lsstGil.h"
#include "lsst/daf/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/fits.h"

namespace lsst { namespace afw { namespace image {

/**
 *  @brief Return the metadata (header entries) from a FITS file.
 *
 *  @deprecated Use lsst::afw::fits::readMetadata instead.
 *
 *  @param[in]    fileName            File to read.
 *  @param[in]    hdu                 HDU to read, 1-indexed.  The special value of 0 will read the
 *                                    first non-empty HDU.
 *  @param[in]    strip               If true, ignore special header keys usually managed by cfitsio
 *                                    (e.g. NAXIS).
 */
inline PTR(daf::base::PropertyList) readMetadata(std::string const & fileName, int hdu=0, bool strip=false) {
    return afw::fits::readMetadata(fileName, hdu, strip);
}

/**
 * Return a value indicating a bad pixel for the given Image type
 *
 * A quiet NaN is returned for types that support it otherwise @c bad
 *
 * @relates lsst::afw::image::Image
 */
template<typename ImageT>
typename ImageT::SinglePixel badPixel(typename ImageT::Pixel bad=0 ///< The bad value if NaN isn't supported
                                     ) {
    typedef typename ImageT::SinglePixel SinglePixelT;
    return SinglePixelT(std::numeric_limits<SinglePixelT>::has_quiet_NaN ?
                        std::numeric_limits<SinglePixelT>::quiet_NaN() : bad);
}

}}} // namespace lsst::afw::image

#endif
