#if !defined(SIMPLE_FITS_H)
#define SIMPLE_FITS_H 1

#include <vw/Image/Manipulation.h>
#include <vw/Image/PixelTypes.h>
#include <vw/Image/ImageResource.h>

// no such file: #include <lsst/pex/utils/Utils.h>
#include <lsst/afw/image/Image.h>
#include <lsst/afw/image/Mask.h>
#include <lsst/afw/image/Wcs.h>

namespace lsst {
namespace afw {
namespace display {

void writeVwFits(int fd,                // file descriptor to write to
                 const vw::ImageBuffer& data, // The data to write
                 const lsst::afw::image::Wcs *Wcs = NULL); // which Wcs to use for pixel

void writeVwFits(const std::string& filename, // file to write, or "| cmd"
                 const vw::ImageBuffer& data, // The data to write
                 const lsst::afw::image::Wcs *Wcs = NULL); // which Wcs to use for pixel

template<typename PIXTYPE>
void writeFits(int fd,                  // file descriptor to write to
               const vw::ImageView<PIXTYPE>& data, // The data to write
               const lsst::afw::image::Wcs *Wcs = NULL   // which Wcs to use for pixel
              ) {
    vw::ImageBuffer buff = data.buffer();
    writeVwFits(fd, buff, Wcs);
}

template<typename PIXTYPE>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const vw::ImageView<PIXTYPE>& data, // The data to write
               const lsst::afw::image::Wcs *Wcs // which Wcs to use for pixel
              ) {
    vw::ImageBuffer buff = data.buffer();
    writeVwFits(filename, buff, Wcs);
}

template<typename MaskPixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const lsst::afw::image::Mask<MaskPixelT>& mask, //!< Mask to write
               const lsst::afw::image::Wcs *Wcs   //!< which Wcs to use for pixel
              ) {
    typename lsst::afw::image::Mask<MaskPixelT>::MaskIVwPtrT vwImagePtr = mask.getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), Wcs);
}

template<typename MaskPixelT>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const lsst::afw::image::Mask<MaskPixelT>& mask, //!< Mask to write
               const lsst::afw::image::Wcs *Wcs   //!< which Wcs to use for pixel
              ) {
    typename lsst::afw::image::Mask<MaskPixelT>::MaskIVwPtrT vwImagePtr = mask.getIVwPtr();
    writeVwFits(filename, vwImagePtr.get()->buffer(), Wcs);
}

               
template<typename ImagePixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const lsst::afw::image::Image<ImagePixelT>& image, //!< Image to write
               const lsst::afw::image::Wcs *Wcs   //!< which Wcs to use for pixel
              ) {
    typename lsst::afw::image::Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), Wcs);
}

template<typename ImagePixelT>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const lsst::afw::image::Image<ImagePixelT>& image, //!< Image to write
               const lsst::afw::image::Wcs *Wcs   //!< which Wcs to use for pixel
              ) {
    typename lsst::afw::image::Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.getIVwPtr();
    writeVwFits(filename, vwImagePtr.get()->buffer(), Wcs);
}

template<typename ImagePixelT>
void writeFits(int fd,                  //!< file descriptor to write to
               const typename lsst::afw::image::Image<ImagePixelT>::ImagePtrT image, //!< Image to write
               const lsst::afw::image::Wcs *Wcs   //!< which Wcs to use for pixel
              ) {
    typename lsst::afw::image::Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.get()->getIVwPtr();
    writeVwFits(fd, vwImagePtr.get()->buffer(), Wcs);
}

template<typename ImagePixelT>
void writeFits(const std::string& filename, // file to write to (or "| cmd")
               const typename lsst::afw::image::Image<ImagePixelT>::ImagePtrT image, //!< Image to write
               const lsst::afw::image::Wcs *Wcs   //!< which Wcs to use for pixel
              ) {
    typename lsst::afw::image::Image<ImagePixelT>::ImageIVwPtrT vwImagePtr = image.get()->getIVwPtr();
    writeVwFits(filename, vwImagePtr.get()->buffer(), Wcs);
}

}}} // namespace lsst::afw::display
#endif
