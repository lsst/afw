/**
 * @file
 *
 * Rotate an Image (or Mask or MaskedImage) by a fixed angle or number of quarter turns
 */
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/offsetImage.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;

namespace lsst {
namespace afw {
namespace math {

template<typename ImageT>
typename ImageT::Ptr binImage(ImageT const& in,  ///< The %image to bin
                              int const binsize, ///< Output pixels are binsize*binsize input pixels
                              lsst::afw::math::Property const flags ///< how to process each pixel
                             )
{
    if (flags != lsst::afw::math::MEAN) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterException,
                          (boost::format("Only afwMath::MEAN is supported, saw 0x%x") % flags).str());
    }
    if (binsize <= 0) {
        throw LSST_EXCEPT(pexExcept::DomainErrorException,
                          (boost::format("Binsize must be >= 0, saw %d") % binsize).str());
    }

    int const outWidth = in.getWidth()/binsize;
    int const outHeight = in.getHeight()/binsize;

    typename ImageT::Ptr out = typename ImageT::Ptr(new ImageT(outWidth, outHeight));
    *out = 0;

    for (int oy = 0, iy = 0; oy < out->getHeight(); ++oy) {
        for (int i = 0; i != binsize; ++i, ++iy) {
            typename ImageT::x_iterator optr = out->row_begin(oy);
            for (typename ImageT::x_iterator iptr = in.row_begin(iy), iend = iptr + binsize*outWidth;
                 iptr < iend; ) {
                typename ImageT::Pixel val = *iptr++;
                for (int j = 1; j != binsize; ++j, ++iptr) {
                    val += *iptr;
                }
                *optr++ += val;
            }
        }
        for (typename ImageT::x_iterator ptr = out->row_begin(oy), end = out->row_end(oy); ptr != end; ++ptr) {
            *ptr /= binsize*binsize;
        }
    }

    return out;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
#define INSTANTIATE(TYPE) \
    template afwImage::Image<TYPE>::Ptr \
             binImage(afwImage::Image<TYPE> const&, int, lsst::afw::math::Property const); \

INSTANTIATE(boost::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)

}}}
