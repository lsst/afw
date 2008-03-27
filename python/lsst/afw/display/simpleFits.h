#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include <vw/Image/Manipulation.h>
#include <vw/Image/PixelTypes.h>
#include <vw/Image/ImageResource.h>

#include <lsst/pex/utils/Utils.h>
#include <lsst/afw/image/Image.h>
#include <lsst/afw/image/Mask.h>
#include <lsst/afw/image/WCS.h>

LSST_START_NAMESPACE(lsst)
LSST_START_NAMESPACE(afw)
LSST_START_NAMESPACE(display)

void writeVwFits(int fd,                // file descriptor to write to
                 const vw::ImageBuffer& data, // The data to write
                 const lsst::afw::image::WCS *WCS = NULL); // which WCS to use for pixel

void writeVwFits(const std::string& filename, // file to write, or "| cmd"
                 const vw::ImageBuffer& data, // The data to write
                 const WCS *WCS = NULL); // which WCS to use for pixel

template<typename PIXTYPE>
void writeFits(int fd,                  // file descriptor to write to
               const vw::ImageView<PIXTYPE>& data, // The data to write
               const WCS *WCS = NULL   // which WCS to use for pixel
              ) {
    vw::ImageBuffer buff = data.buffer();
    writeVwFits(fd, buff, WCS);
}

template<typename PIXTYPE>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const vw::ImageView<PIXTYPE>& data, // The data to write
               const WCS *WCS // which WCS to use for pixel
              ) {
    vw::ImageBuffer buff = data.buffer();
    writeVwFits(filename, buff, WCS);
}

template<typename MaskPixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const lsst::afw::image::Mask<MaskPixelT>& mask, //!< Mask to write
               const WCS *WCS   //!< which WCS to use for pixel
              ) {
    typename Mask<MaskPixelT>::MaskIVwPtrT vwImagePtr = mask.getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), WCS);
}

template<typename MaskPixelT>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const lsst::afw::image::Mask<MaskPixelT>& mask, //!< Mask to write
               const WCS *WCS   //!< which WCS to use for pixel
              ) {
    typename Mask<MaskPixelT>::MaskIVwPtrT vwImagePtr = mask.getIVwPtr();
    writeVwFits(filename, vwImagePtr.get()->buffer(), WCS);
}

               
template<typename ImagePixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const lsst::afw::image::Image<ImagePixelT>& image, //!< Image to write
               const WCS *WCS   //!< which WCS to use for pixel
              ) {
    typename Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), WCS);
}

template<typename ImagePixelT>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const lsst::afw::image::Image<ImagePixelT>& image, //!< Image to write
               const WCS *WCS   //!< which WCS to use for pixel
              ) {
    typename Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.getIVwPtr();
    writeVwFits(filename, vwImagePtr.get()->buffer(), WCS);
}

template<typename ImagePixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const typename lsst::afw::image::Image<ImagePixelT>::ImagePtrT image, //!< Image to write
               const WCS *WCS   //!< which WCS to use for pixel
              ) {
    typename Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.get()->getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), WCS);
}

template<typename ImagePixelT>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const typename lsst::afw::image::Image<ImagePixelT>::ImagePtrT image, //!< Image to write
               const WCS *WCS   //!< which WCS to use for pixel
              ) {
    typename Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.get()->getIVwPtr();
    writeVwFits(filename, vwImagePtr.get()->buffer(), WCS);
}

LSST_END_NAMESPACE(display)
LSST_END_NAMESPACE(afw)
LSST_END_NAMESPACE(lsst)
#endif
