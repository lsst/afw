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
 
#ifndef LSST_AFW_IMAGE_fits_io_h_INCLUDED
#define LSST_AFW_IMAGE_fits_io_h_INCLUDED

#include <string>

#include "boost/format.hpp"
#include "lsst/pex/exceptions.h"
#include "lsst/afw/fits.h"
#include "lsst/afw/geom.h"
#include "lsst/afw/image/Wcs.h"
#include "ndarray.h"

namespace lsst { namespace afw { namespace image {

template <typename PixelT>
inline void fits_read_array(
    fits::Fits & fitsfile,
    ndarray::Array<PixelT,2,2> & array,
    geom::Point2I & xy0,
    lsst::daf::base::PropertySet & metadata,
    geom::Box2I bbox=geom::Box2I(),
    ImageOrigin origin=PARENT
) {
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
        if (origin == PARENT) {
            bbox.shift(-xyOffset);
        }
        xy0 = bbox.getMin();

        if (bbox.getMinX() < 0 || bbox.getMinY() < 0 ||
            bbox.getWidth() > dimensions.getX() || bbox.getHeight() > dimensions.getY()
        ) {
            throw LSST_EXCEPT(
                lsst::pex::exceptions::LengthError,
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

template <typename ImageT>
inline void fits_write_image(
    fits::Fits & fitsfile, const ImageT & image,
    CONST_PTR(daf::base::PropertySet) metadata=CONST_PTR(daf::base::PropertySet)()
) {
    fitsfile.createImage<typename ImageT::Pixel>(image.getArray().getShape());
    if (metadata) {
        fitsfile.writeMetadata(*metadata);
    }
    fitsfile.writeImage(image.getArray());
}

}}}                                     // namespace lsst::afw::image

#endif // !LSST_AFW_IMAGE_fits_io_h_INCLUDED
