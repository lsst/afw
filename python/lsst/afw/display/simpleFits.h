/**
 * \file
 * \brief Definitions to write a FITS image
 */
#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/image/Wcs.h"

namespace lsst {
namespace afw {
namespace display {

template<typename ImageT>
void writeBasicFits(int fd, ImageT const& data, lsst::afw::image::Wcs const* Wcs=NULL, char const* title=NULL);

template<typename ImageT>
void writeBasicFits(std::string const& filename, ImageT const& data, lsst::afw::image::Wcs const* Wcs=NULL,
                    const char* title=NULL);

}}} // namespace lsst::afw::display
#endif
