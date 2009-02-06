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
#include "lsst/pex/exceptions.h"
#include "fits_io_private.h"

namespace lsst { namespace afw { namespace image {

/// \ingroup FITS_IO
/// \brief Determines whether the given view type is supported for reading
template <typename View>
struct fits_read_support {
    BOOST_STATIC_CONSTANT(bool, is_supported = detail::fits_read_support_private<View>::is_supported);
    BOOST_STATIC_CONSTANT(int,  BITPIX = detail::fits_read_support_private<View>::BITPIX);
    BOOST_STATIC_CONSTANT(bool, value = is_supported);
};

/// \ingroup FITS_IO
/// \brief Returns the width and height of the FITS file at the specified location.
/// Throws lsst::afw::image::FitsException if the location does not correspond to a valid FITS file
inline boost::gil::point2<std::ptrdiff_t> fits_read_dimensions(const char* filename) {
    lsst::daf::base::PropertySet::Ptr metadata(new lsst::daf::base::PropertySet());
    detail::fits_reader m(filename, metadata);
    return m.get_getDimensions();
}

/// \ingroup FITS_IO
/// \brief Returns the width and height of the FITS file at the specified location.
/// Throws lsst::afw::image::FitsException if the location does not correspond to a valid FITS file
inline boost::gil::point2<std::ptrdiff_t> fits_read_dimensions(const std::string& filename) {
    return fits_read_dimensions(filename.c_str());
}

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

/// \ingroup FITS_IO
/// \brief Allocates a new image whose dimensions are determined by the given fits image file, and loads the
/// pixels into it.
///
/// Triggers a compile assert if the image channel depth is not supported by the FITS library or by the I/O
/// extension.  Throws lsst::afw::image::FitsException if the file is not a valid FITS file, or
/// if its color space or channel depth are not compatible with the ones specified by Image
template <typename Image>
inline void fits_read_image(const std::string& filename, Image& im,
                            lsst::daf::base::PropertySet::Ptr metadata = lsst::daf::base::PropertySet::Ptr()
                           ) {
    BOOST_STATIC_ASSERT(fits_read_support<typename Image::view_t>::is_supported);

    detail::fits_reader m(filename, metadata);
    m.read_image(im);
}

/// \ingroup FITS_IO
/// \brief Saves the view to a fits file specified by the given fits image file name.
/// Triggers a compile assert if the view channel depth is not supported by the FITS library or by the I/O extension.
/// Throws lsst::afw::image::FitsException if it fails to create the file.
template <typename View>
inline void fits_write_view(const std::string& filename, const View& view,
                            lsst::daf::base::PropertySet::Ptr metadata = lsst::daf::base::PropertySet::Ptr()
                           ) {
    BOOST_STATIC_ASSERT(fits_read_support<View>::is_supported);

    detail::fits_writer m(filename);
    m.apply(view, metadata);
}

}}}                                     // namespace lsst::afw::image
/// \endcond
#endif
