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
 * \brief  Internal support for reading and writing FITS files
 *
 * Tell doxygen to (usually) ignore this file \cond GIL_IMAGE_INTERNALS
 * \author Robert Lupton (rhl@astro.princeton.edu)
 *         Princeton University
 * \date   September 2008
 */
#if !defined(LSST_FITS_IO_H)
#define LSST_FITS_IO_H

#include <cstdio>
#include <algorithm>
#include <string>
#include "boost/static_assert.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/format.hpp"
#include "boost/gil/extension/io/io_error.hpp"
#include "lsst/afw/image/lsstGil.h"
#include "lsst/pex/exceptions.h"
#include "fits_io_private.h"
#include "lsst/afw/fits.h"
#include "ndarray.h"

namespace lsst { namespace afw { namespace image {

/// \ingroup FITS_IO
/// \brief Determines whether the given view type is supported for reading
template <typename PixelT>
struct fits_read_support {
#if !defined(SWIG)
    BOOST_STATIC_CONSTANT(bool, is_supported = detail::fits_read_support_private<PixelT>::is_supported);
    BOOST_STATIC_CONSTANT(int,  BITPIX = detail::fits_read_support_private<PixelT>::BITPIX);
    BOOST_STATIC_CONSTANT(bool, value = is_supported);
#endif
};

/// \ingroup FITS_IO
/// \brief Returns the width and height of the FITS file at the specified location.
/// Throws lsst::afw::image::FitsException if the location does not correspond to a valid FITS file
inline geom::Extent2I fits_read_dimensions(const char* filename) {
    lsst::daf::base::PropertySet metadata;
    detail::fits_reader m(filename, metadata);
    return m.getDimensions();
}

/// \ingroup FITS_IO
/// \brief Returns the width and height of the FITS file at the specified location.
/// Throws lsst::afw::image::FitsException if the location does not correspond to a valid FITS file
inline geom::Extent2I fits_read_dimensions(const std::string& filename) {
    return fits_read_dimensions(filename.c_str());
}

template <typename PixelT>
inline void fits_read_array(
    fits::Fits & fitsfile,
    ndarray::Array<PixelT,2,2> & array,
    geom::Point2I & xy0,
    lsst::daf::base::PropertySet & metadata,
    geom::Box2I bbox = geom::Box2I(),
    ImageOrigin origin = LOCAL
) {
    BOOST_STATIC_ASSERT(fits_read_support<PixelT>::is_supported);

    if (!fitsfile.checkImageType<PixelT>()) {
        throw LSST_FITS_EXCEPT(
            fits::FitsTypeError,
            fitsfile,
            "Incorrect image type for FITS image"
        );  
    }

    int nAxis = fitsfile.getImageDim();
    ndarray::Vector<int,2> shape;
    if (nAxis == 2) {
        shape = fitsfile.getImageShape<2>();
    } else if (nAxis == 3) {
        ndarray::Vector<int,3> shape3 = fitsfile.getImageShape<3>();
        if (shape3[0] != 1) {
            throw LSST_EXCEPT(
                fits::FitsError,
                boost::str(boost::format("3rd dimension %d is not 1") % shape3[0])
            );
        }
        shape = shape3.last<2>();
    }

    fitsfile.readMetadata(metadata, true);

    // Origin of part of image to read
    xy0 = geom::Point2I();

    geom::Extent2I xyOffset(detail::getImageXY0FromMetadata(detail::wcsNameForXY0, &metadata));
    geom::Extent2I dimensions = geom::Extent2I(shape[1], shape[0]);

    if (!bbox.isEmpty()) {
        if(origin == PARENT) {
            bbox.shift(-xyOffset);
        }
        xy0 = bbox.getMin();

        if (bbox.getMinX() < 0 || bbox.getMinY() < 0 ||
            bbox.getWidth() > dimensions.getX() || bbox.getHeight() > dimensions.getY()
        ) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthErrorException,
                (boost::format("BBox (%d,%d) %dx%d doesn't fit in image %dx%d") %
                 bbox.getMinX() % bbox.getMinY() % bbox.getWidth() % bbox.getHeight() %
                 dimensions.getX() % dimensions.getY()).str()
            );
        }
        dimensions = bbox.getDimensions();
    }

    array = ndarray::allocate(dimensions.getY(), dimensions.getX());
    
    fitsfile.readImage(array, ndarray::makeVector(xy0.getY(), xy0.getX()));

    xy0 += xyOffset;
}

/// \ingroup FITS_IO
/// \brief Saves the view to a fits file specified by the given fits image file name.
/// Triggers a compile assert if the view channel depth is not supported by the FITS library or by the I/O extension.
/// Throws lsst::afw::image::FitsException if it fails to create the file.
template <typename ImageT>
inline void fits_write_image(
    fits::Fits & fitsfile, const ImageT & image,
    CONST_PTR(daf::base::PropertySet) metadata = CONST_PTR(daf::base::PropertySet)()
) {
    BOOST_STATIC_ASSERT(fits_read_support<typename ImageT::Pixel>::is_supported);
    fitsfile.createImage<typename ImageT::Pixel>(image.getArray().getShape());

    if (metadata) {
        fitsfile.writeMetadata(*metadata);
    }
    fitsfile.writeImage(image.getArray());
}

}}}                                     // namespace lsst::afw::image
/// \endcond
#endif
