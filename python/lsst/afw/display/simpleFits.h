#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst {
namespace afw {
namespace display {

template<typename ImageT>
void writeBasicFits(int fd,                // file descriptor to write to
                    ImageT const& data,    // The data to write
                    lsst::afw::image::Wcs const* Wcs = NULL); // which Wcs to use for pixel

template<typename ImageT>
void writeBasicFits(std::string const& filename, // file to write, or "| cmd"
                    ImageT const& data,          // The data to write
                    lsst::afw::image::Wcs const* Wcs = NULL); // which Wcs to use for pixel

}}} // namespace lsst::afw::display
#endif
