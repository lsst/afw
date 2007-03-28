#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include <vw/Image/Manipulation.h>
#include <vw/Image/PixelTypes.h>
#include <vw/Image/ImageResource.h>

#include "lsst/fw/Utils.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

void writeFits(int fd,                // file descriptor to write to
               vw::ImageBuffer &data, // The data to write
               const std::string &WCS); // which WCS to use for pixel

void writeFitsFile(const std::string &filename, // file to write, or "| cmd"
                   vw::ImageBuffer &data, // The data to write
                   const std::string &WCS); // which WCS to use for pixel

template<typename PIXTYPE>
void writeFits(int fd,               // file descriptor to write to
               vw::ImageView<PIXTYPE> &data, // The data to write
               const std::string &WCS // which WCS to use for pixel
              ) {
    vw::ImageBuffer buff = data.buffer();
    writeFits(fd, buff, WCS);
}

template<typename PIXTYPE>
void writeFitsFile(const std::string &filename, // file to write to (or "| cmd")
                   vw::ImageView<PIXTYPE> &data, // The data to write
                   const std::string &WCS // which WCS to use for pixel
                  ) {
    vw::ImageBuffer buff = data.buffer();
    writeFitsFile(filename, buff, WCS);
}

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
