/// \file
/// Tell doxygen to (usually) ignore this file \cond GIL_IMAGE_INTERNALS
/// \brief  Internal support for reading and writing FITS files
/// \author Robert Lupton (rhl@astro.princeton.edu)
///         Princeton University
/// \date   September 2008
/// \note Requires cfitsio

/*************************************************************************************************/

#if !defined(LSST_FITS_DYNAMIC_IO_H)
#define LSST_FITS_DYNAMIC_IO_H

#include <stdio.h>
#include <string>
#include <boost/mpl/bool.hpp>
#include <boost/shared_ptr.hpp>
#include "boost/gil/extension/dynamic_image/dynamic_image_all.hpp"
#include "boost/gil/extension/io/io_error.hpp"
#include "boost/gil/extension/io/dynamic_io.hpp"

#include "fits_io.h"
#include "fits_io_private.h"

namespace lsst { namespace afw { namespace image {

namespace detail {
    
class fits_type_format_checker {
    int _bitpix;
public:
    fits_type_format_checker(int bitpix) : _bitpix(bitpix) {}
    
    template <typename Image>
    bool apply() {
        return fits_read_support<typename Image::view_t>::BITPIX == _bitpix;
    }
};

struct fits_read_is_supported {
    template<typename View> struct apply
        : public boost::mpl::bool_<fits_read_support<View>::is_supported> {};
};

class fits_reader_dynamic : public fits_reader {
public:
    fits_reader_dynamic(cfitsio::fitsfile* file)     : fits_reader(file) {}
    fits_reader_dynamic(const std::string& filename) : fits_reader(filename) {}
        
    template <typename Images>
    void read_image(boost::gil::any_image<Images>& im) {
        if (!construct_matched(im,detail::fits_type_format_checker(_bitpix))) {
            throw lsst::pex::exceptions::FitsError("fits_reader_dynamic::read_image(): "
                                                   "no matching image type between those of the given any_image and that of the file");
        } else {
            im.recreate(get_dimensions());
            boost::gil::detail::dynamic_io_fnobj<fits_read_is_supported, fits_reader> op(this);
            apply_operation(view(im),op);
        }
    }
};

} // namespace detail


/// \ingroup FITS_IO
/// \brief reads a FITS image into a run-time instantiated image
/// Opens the given FITS file name, selects the first type in Images whose color space and channel are compatible to those of the image file
/// and creates a new image of that type with the dimensions specified by the image file.
/// Throws lsst::pex::exceptions::FitsError if none of the types in Images are compatible with the type on disk.
template <typename Images>
inline void fits_read_image(const char* filename, boost::gil::any_image<Images>& im) {
    detail::fits_reader_dynamic m(filename);
    m.read_image(im);
}

/// \ingroup FITS_IO
/// \brief reads a FITS image into a run-time instantiated image
template <typename Images>
inline void fits_read_image(const std::string& filename, boost::gil::any_image<Images>& im) {
    fits_read_image(filename.c_str(),im);
}

}}}                                     // namespace lsst::afw::image
/// \endcond
#endif
