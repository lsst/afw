/**
 * @file
 *
 * Rotate an Image (or Mask or MaskedImage) by a fixed angle or number of quarter turns
 */
#include "lsst/afw/math/offsetImage.h"

namespace afwImage = lsst::afw::image;

namespace lsst {
namespace afw {
namespace math {

/**
 * Rotate an image by an integral number of quarter turns
 */
template<typename ImageT>
typename ImageT::Ptr rotateImageBy90(ImageT const& inImage, ///< The %image to rotate
                                     int nQuarter ///< the desired number of quarter turns
                                    ) {
    typename ImageT::Ptr outImage;      // output image

    switch (nQuarter%4) {
      case 0:
        outImage.reset(new ImageT(inImage, true)); // a deep copy of inImage
        break;
      case 1:
        outImage.reset(new ImageT(inImage.getHeight(), inImage.getWidth()));

        for (int y = 0; y != inImage.getHeight(); ++y) {
            typename ImageT::y_iterator optr = outImage->col_begin(inImage.getHeight() - y - 1);
            for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                 iptr != end; ++iptr, ++optr) {
                *optr = *iptr;
            }
        }
        
        break;
      case 2:
        outImage.reset(new ImageT(inImage.getDimensions()));
        
        for (int y = 0; y != inImage.getHeight(); ++y) {
            typename ImageT::x_iterator optr = outImage->row_begin(inImage.getHeight() - y - 1);
            int x = inImage.getWidth() - 1;
            for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                 iptr != end; ++iptr, --x) {
                optr[x] = *iptr;
            }
        }
        break;
      case 3:
        outImage.reset(new ImageT(inImage.getHeight(), inImage.getWidth()));

        for (int y = 0; y != inImage.getHeight(); ++y) {
            typename ImageT::y_iterator optr = outImage->col_begin(y);
            int x = inImage.getWidth() - 1;
            for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                 iptr != end; ++iptr, --x) {
                optr[x] = *iptr;
            }
        }
        
        break;
    }

    return outImage;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
#define INSTANTIATE(TYPE) \
    template afwImage::Image<TYPE>::Ptr rotateImageBy90(afwImage::Image<TYPE> const&, int); \
    //template afwImage::MaskedImage<TYPE>::Ptr rotateImageBy90(afwImage::MaskedImage<TYPE> const&, int);

INSTANTIATE(boost::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)

}}}
