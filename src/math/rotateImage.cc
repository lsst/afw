/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * @file
 *
 * Rotate an Image (or Mask or MaskedImage) by a fixed angle or number of quarter turns
 */
#include "lsst/afw/math/offsetImage.h"

namespace afwImage = lsst::afw::image;
namespace afwGeom = lsst::afw::geom;

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

    while (nQuarter < 0) {
        nQuarter += 4;
    }

    switch (nQuarter%4) {
    case 0:
        outImage.reset(new ImageT(inImage, true)); // a deep copy of inImage
        break;
    case 1:
        outImage.reset(new ImageT(afwGeom::Extent2I(inImage.getHeight(), inImage.getWidth())));
                       
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
            typename ImageT::x_iterator optr = outImage->x_at(inImage.getWidth() - 1,
                                                              inImage.getHeight() - y - 1);
            for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                 iptr != end; ++iptr, optr -= 1) {
                *optr = *iptr;
            }
        }
        break;
    case 3:
        outImage.reset(new ImageT(afwGeom::Extent2I(inImage.getHeight(), inImage.getWidth())));

        for (int y = 0; y != inImage.getHeight(); ++y) {
            typename ImageT::y_iterator optr = outImage->y_at(y, inImage.getWidth() - 1);
            for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                 iptr != end; ++iptr, optr -= 1) {
                *optr = *iptr;
            }
        }
        
        break;
    }

    return outImage;
}

/**
 * Flip an image left--right and/or top--bottom
 */
template<typename ImageT>
PTR(ImageT) flipImage(ImageT const& inImage, ///< The %image to flip
                      bool flipLR,           ///< Flip left <--> right?
                      bool flipTB            ///< Flip top <--> bottom?
                     ) {
    typename ImageT::Ptr outImage(new ImageT(inImage, true)); // Output image

    if (flipLR) {
        if (flipTB) {
            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::x_iterator optr = outImage->x_at(inImage.getWidth() - 1,
                                                                  inImage.getHeight() - y - 1);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, optr -= 1) {
                    *optr = *iptr;
                }
            }
        } else {
            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::x_iterator optr = outImage->x_at(inImage.getWidth() - 1, y);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, optr -= 1) {
                    *optr = *iptr;
                }
            }
        }
    } else {
        if (flipTB) {
            for (int y = 0; y != inImage.getHeight(); ++y) {
                typename ImageT::x_iterator optr = outImage->row_begin(inImage.getHeight() - y - 1);
                for (typename ImageT::x_iterator iptr = inImage.row_begin(y), end = inImage.row_end(y);
                     iptr != end; ++iptr, ++optr) {
                    *optr = *iptr;
                }
            }
        } else {
            ;                           // nothing to do
        }
    }

    return outImage;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
/// \cond
#define INSTANTIATE(TYPE) \
    template afwImage::Image<TYPE>::Ptr rotateImageBy90(afwImage::Image<TYPE> const&, int); \
    template afwImage::MaskedImage<TYPE>::Ptr rotateImageBy90(afwImage::MaskedImage<TYPE> const&, int); \
    template afwImage::Image<TYPE>::Ptr flipImage(afwImage::Image<TYPE> const&, bool flipLR, bool flipTB); \
    template afwImage::MaskedImage<TYPE>::Ptr flipImage(afwImage::MaskedImage<TYPE> const&, bool flipLR, bool flipTB); 
    

INSTANTIATE(boost::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)
template afwImage::Mask<boost::uint16_t>::Ptr rotateImageBy90(afwImage::Mask<boost::uint16_t> const&, int); 
template afwImage::Mask<boost::uint16_t>::Ptr flipImage(afwImage::Mask<boost::uint16_t> const&, bool flipLR, bool flipTB); 
/// \endcond

}}}
