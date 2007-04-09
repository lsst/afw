#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include <vw/Image/Manipulation.h>
#include <vw/Image/PixelTypes.h>
#include <vw/Image/ImageResource.h>

#include "lsst/fw/Utils.h"
#include "lsst/fw/Image.h"
#include "lsst/fw/Mask.h"

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(fw)

void writeVwFits(int fd,                // file descriptor to write to
                 const vw::ImageBuffer& data, // The data to write
                 const std::string& WCS); // which WCS to use for pixel

void writeVwFitsFile(const std::string& filename, // file to write, or "| cmd"
                     const vw::ImageBuffer& data, // The data to write
                     const std::string& WCS); // which WCS to use for pixel

template<typename PIXTYPE>
void writeFits(int fd,               // file descriptor to write to
               const vw::ImageView<PIXTYPE>& data, // The data to write
               const std::string& WCS // which WCS to use for pixel
              ) {
    vw::ImageBuffer buff = data.buffer();
    writeVwFits(fd, buff, WCS);
}

template<typename PIXTYPE>
void writeFitsFile(const std::string& filename, // file to write to (or "| cmd")
                   const vw::ImageView<PIXTYPE>& data, // The data to write
                   const std::string& WCS // which WCS to use for pixel
                  ) {
    vw::ImageBuffer buff = data.buffer();
    writeVwFitsFile(filename, buff, WCS);
}

template<typename MaskPixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const lsst::Mask<MaskPixelT>& mask, //!< Mask to write
               const std::string& WCS   //!< which WCS to use for pixel
              ) {
    typename Mask<MaskPixelT>::MaskIVwPtrT vwImagePtr = mask.getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), WCS);
}

template<typename MaskPixelT>
void writeFitsFile(const std::string& filename, // file to write to (or "| cmd")
                   const lsst::Mask<MaskPixelT>& mask, //!< Mask to write
                   const std::string& WCS   //!< which WCS to use for pixel
              ) {
    typename Mask<MaskPixelT>::MaskIVwPtrT vwImagePtr = mask.getIVwPtr();
    writeVwFitsFile(filename, vwImagePtr.get()->buffer(), WCS);
}

               
template<typename ImagePixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const lsst::Image<ImagePixelT>& image, //!< Image to write
               const std::string& WCS   //!< which WCS to use for pixel
              ) {
    typename Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), WCS);
}

template<typename ImagePixelT>
void writeFitsFile(const std::string& filename, // file to write to (or "| cmd")
                   const lsst::Image<ImagePixelT>& image, //!< Image to write
                   const std::string& WCS   //!< which WCS to use for pixel
              ) {
    typename Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.getIVwPtr();
    writeVwFitsFile(filename, vwImagePtr.get()->buffer(), WCS);
}

#if 0                                   // this makes _wrap.cc fail to compile
template<typename ImagePixelT>
void writeFitsFile(const std::string& filename, // file to write to (or "| cmd")
                   const typename lsst::Image<ImagePixelT>::ImagePtrT image, //!< Image to write
                   const std::string& WCS   //!< which WCS to use for pixel
              ) {
    typename Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.get()->getIVwPtr();
    writeVwFitsFile(filename, vwImagePtr.get()->buffer(), WCS);
}
#endif

LSST_END_NAMESPACE(fw)
LSST_END_NAMESPACE(lsst)
#endif
