// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
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
#include "boost/shared_ptr.hpp"

#include "lsst/afw/image/lsstGil.h"
#include "lsst/daf/base.h"
#include "lsst/daf/base/Citizen.h"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/formatters/ImageFormatter.h"

namespace lsst { namespace afw { namespace image {

/**
 *  @brief Return the metadata (header entries) from a FITS file.
 *
 *  @param[in]    fileName            File to read.
 *  @param[in]    hdu                 HDU to read, 1-indexed.  The special value of 0 will read the
 *                                    first non-empty HDU.
 *  @param[in]    strip               If true, ignore special header keys usually managed by cfitsio
 *                                    (e.g. NAXIS).
 */
PTR(daf::base::PropertySet) readMetadata(std::string const & fileName, int hdu=0, bool strip=false);

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
