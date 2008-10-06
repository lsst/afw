#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst {
namespace afw {
namespace display {

template<typename PixelT>
void writeBasicFits(int fd,                                      // file descriptor to write to
                    lsst::afw::image::Image<PixelT> const& data, // The data to write
                    lsst::afw::image::Wcs const* Wcs = NULL);    // which Wcs to use for pixel

template<typename PixelT>
void writeBasicFits(std::string const& filename,                 // file to write, or "| cmd"
                    lsst::afw::image::Image<PixelT> const& data, // The data to write
                    lsst::afw::image::Wcs const* Wcs = NULL);    // which Wcs to use for pixel

}}} // namespace lsst::afw::display
#endif
