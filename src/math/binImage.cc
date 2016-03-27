/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
 
/**
 * @file
 *
 * Bin an Image or MaskedImage by an integral factor (the same in x and y)
 */
#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/offsetImage.h"

namespace pexExcept = lsst::pex::exceptions;
namespace afwImage = lsst::afw::image;

namespace lsst {
namespace afw {
namespace math {

template<typename ImageT>
PTR(ImageT) binImage(ImageT const& in,  ///< The %image to bin
                     int const binsize, ///< Output pixels are binsize*binsize input pixels
                     lsst::afw::math::Property const flags ///< how to generate super-pixels
                    )
{
    return binImage(in, binsize, binsize, flags);
}

template<typename ImageT>
PTR(ImageT) binImage(ImageT const& in,  ///< The %image to bin
                     int const binX,    ///< Output pixels are binX*binY input pixels
                     int const binY,    ///< Output pixels are binX*binY input pixels
                     lsst::afw::math::Property const flags ///< how to generate super-pixels
                    )
{
    if (flags != lsst::afw::math::MEAN) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          (boost::format("Only afwMath::MEAN is supported, saw 0x%x") % flags).str());
    }
    if (binX <= 0 || binY <= 0) {
        throw LSST_EXCEPT(pexExcept::DomainError,
                          (boost::format("Binning must be >= 0, saw %dx%d") % binX % binY).str());
    }

    int const outWidth = in.getWidth()/binX;
    int const outHeight = in.getHeight()/binY;

    typename ImageT::Ptr out = typename ImageT::Ptr(
        new ImageT(geom::Extent2I(outWidth, outHeight))
    );
    out->setXY0(in.getXY0());
    *out = typename ImageT::SinglePixel(0);

    for (int oy = 0, iy = 0; oy < out->getHeight(); ++oy) {
        for (int i = 0; i != binY; ++i, ++iy) {
            typename ImageT::x_iterator optr = out->row_begin(oy);
            for (typename ImageT::x_iterator iptr = in.row_begin(iy), iend = iptr + binX*outWidth;
                 iptr < iend; ) {
                typename ImageT::SinglePixel val = *iptr; ++iptr;
                for (int j = 1; j != binX; ++j, ++iptr) {
                    val += *iptr;
                }
                *optr += val; ++optr;
            }
        }
        for (typename ImageT::x_iterator ptr = out->row_begin(oy), end = out->row_end(oy); ptr != end; ++ptr) {
            *ptr /= binX*binY;
        }
    }

    return out;
}

/************************************************************************************************************/
//
// Explicit instantiations
//
/// \cond
#define INSTANTIATE(TYPE) \
    template afwImage::Image<TYPE>::Ptr \
             binImage(afwImage::Image<TYPE> const&, int, lsst::afw::math::Property const); \
    template afwImage::Image<TYPE>::Ptr \
             binImage(afwImage::Image<TYPE> const&, int, int, lsst::afw::math::Property const); \
    template afwImage::MaskedImage<TYPE>::Ptr \
             binImage(afwImage::MaskedImage<TYPE> const&, int, lsst::afw::math::Property const); \
    template afwImage::MaskedImage<TYPE>::Ptr \
             binImage(afwImage::MaskedImage<TYPE> const&, int, int, lsst::afw::math::Property const); \

INSTANTIATE(boost::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)
/// \endcond

}}}
