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
#include <boost/static_assert.hpp>
#include <boost/shared_ptr.hpp>
#include "boost/gil/extension/io/io_error.hpp"
#include "lsst/afw/image/lsstGil.h"
#include "lsst/pex/exceptions.h"
#include "fits_io_private.h"

#include "lsst/ndarray.h"

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
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertyList());
    detail::fits_reader m(filename, metadata);
    return m.getDimensions();
}

/// \ingroup FITS_IO
/// \brief Returns the width and height of the FITS file at the specified location.
/// Throws lsst::afw::image::FitsException if the location does not correspond to a valid FITS file
inline geom::Extent2I fits_read_dimensions(const std::string& filename) {
    return fits_read_dimensions(filename.c_str());
}

#if 0
/// \ingroup FITS_IO
/// \brief Loads the image specified by the given fits image file name into the given view.
/// Triggers a compile assert if the view channel depth is not supported by the FITS library or by the I/O
/// extension.  Throws lsst::afw::image::FitsException if the file is not a valid FITS file, or
/// if its color space or channel depth are not compatible with the ones specified by View, or if its
/// dimensions don't match the ones of the view.
template <typename View>
inline void fits_read_view(std::string const& filename,const View& view,
                           lsst::daf::base::PropertySet::Ptr metadata = lsst::daf::base::PropertySet::Ptr()
                          ) {
    BOOST_STATIC_ASSERT(fits_read_support<View>::is_supported);

    detail::fits_reader m(filename, metadata);
    m.apply(view);
}
#endif

/// \ingroup FITS_IO
/// \brief Allocates a new image whose dimensions are determined by the given fits image file, and loads the
/// pixels into it.
///
/// Triggers a compile assert if the image channel depth is not supported by the FITS library or by the I/O
/// extension.  Throws lsst::afw::image::FitsException if the file is not a valid FITS file, or
/// if its color space or channel depth are not compatible with the ones specified by Image
 template <typename PixelT>
 inline void fits_read_image(const std::string& filename,
                             lsst::ndarray::Array<PixelT,2,2> & array,
                             geom::Point2I & xy0,
                             lsst::daf::base::PropertySet::Ptr metadata = lsst::daf::base::PropertySet::Ptr(),
                             int hdu=1,
                             geom::Box2I const& bbox=geom::Box2I(),
                             ImageOrigin const origin = LOCAL
 ) {
    BOOST_STATIC_ASSERT(fits_read_support<PixelT>::is_supported);

    detail::fits_reader m(filename, metadata, hdu, bbox, origin);
    m.read_image(array, xy0);
}

/// \ingroup FITS_IO
/// \brief Allocates a new image whose dimensions are determined by the given fits image RAM-file, and loads the
/// pixels into it.
///
/// Triggers a compile assert if the image channel depth is not supported by the FITS library or by the I/O
/// extension.  Throws lsst::afw::image::FitsException if the RAM-file is not a valid FITS file, or
/// if its color space or channel depth are not compatible with the ones specified by Image
 template <typename PixelT>
 inline void fits_read_ramImage(char **ramFile, size_t *ramFileLen,
                             lsst::ndarray::Array<PixelT,2,2> & array,
                             geom::Point2I & xy0,
                             lsst::daf::base::PropertySet::Ptr metadata = lsst::daf::base::PropertySet::Ptr(),
                             int hdu=1,
                             geom::Box2I const& bbox=geom::Box2I(),
                             ImageOrigin const origin = LOCAL
 ) {
    BOOST_STATIC_ASSERT(fits_read_support<PixelT>::is_supported);

    detail::fits_reader m(ramFile, ramFileLen, metadata, hdu, bbox, origin);
    m.read_image(array, xy0);
}

/// \ingroup FITS_IO
/// \brief Saves the view to a fits file specified by the given fits image file name.
/// Triggers a compile assert if the view channel depth is not supported by the FITS library or by the I/O extension.
/// Throws lsst::afw::image::FitsException if it fails to create the file.
template <typename ImageT>
inline void fits_write_image(const std::string& filename, const ImageT & image,
                            boost::shared_ptr<const lsst::daf::base::PropertySet> metadata = lsst::daf::base::PropertySet::Ptr(),
                            std::string const& mode="w"
                           ) {
    BOOST_STATIC_ASSERT(fits_read_support<typename ImageT::Pixel>::is_supported);

    detail::fits_writer m(filename, mode);
    m.apply(image, metadata);
}

/// \ingroup FITS_IO
/// \brief Saves the view to a fits RAM-file.
/// Triggers a compile assert if the view channel depth is not supported by the FITS library or by the I/O extension.
/// Throws lsst::afw::image::FitsException if it fails to create the RAM-file.
template <typename ImageT>
inline void fits_write_ramImage(char **ramFile, size_t *ramFileLen, const ImageT & image,
                            boost::shared_ptr<const lsst::daf::base::PropertySet> metadata = lsst::daf::base::PropertySet::Ptr(),
                            std::string const& mode="w"
                           ) {
    BOOST_STATIC_ASSERT(fits_read_support<typename ImageT::Pixel>::is_supported);

    detail::fits_writer m(ramFile, ramFileLen, mode);
    m.apply(image, metadata);
}

}}}                                     // namespace lsst::afw::image
/// \endcond
#endif
