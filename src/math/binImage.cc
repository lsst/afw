/* 
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
 * 
 * This product includes software developed by the
 * LSST Project (http://www.lsst.org/).
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the LSST License Statement and 
 * the GNU General Public License along with this program.  If not, 
 * see <http://www.lsstcorp.org/LegalNotices/>.
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
typename ImageT::Ptr binImage(ImageT const& in,  ///< The %image to bin
                              int const binsize, ///< Output pixels are binsize*binsize input pixels
                              lsst::afw::math::Property const flags ///< how to generate super-pixels
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

    typename ImageT::Ptr out = typename ImageT::Ptr(
        new ImageT(geom::ExtentI(outWidth, outHeight))
    );
    *out = typename ImageT::SinglePixel(0);

    for (int oy = 0, iy = 0; oy < out->getHeight(); ++oy) {
        for (int i = 0; i != binsize; ++i, ++iy) {
            typename ImageT::x_iterator optr = out->row_begin(oy);
            for (typename ImageT::x_iterator iptr = in.row_begin(iy), iend = iptr + binsize*outWidth;
                 iptr < iend; ) {
                typename ImageT::SinglePixel val = *iptr; ++iptr;
                for (int j = 1; j != binsize; ++j, ++iptr) {
                    val += *iptr;
                }
                *optr += val; ++optr;
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
    template afwImage::MaskedImage<TYPE>::Ptr \
             binImage(afwImage::MaskedImage<TYPE> const&, int, lsst::afw::math::Property const); \

INSTANTIATE(boost::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)

}}}
