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

/*
 * Bin an Image or MaskedImage by an integral factor (the same in x and y)
 */
#include <memory>
#include <cstdint>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/offsetImage.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace math {

template <typename ImageT>
std::shared_ptr<ImageT> binImage(ImageT const& in, int const binsize, lsst::afw::math::Property const flags) {
    return binImage(in, binsize, binsize, flags);
}

namespace {

// Bin a single image or variance plane using a double accumulator to prevent
// integer overflow when summing many pixels of a narrow type (e.g. uint16_t).
template <typename PixelT>
std::shared_ptr<image::Image<PixelT>> _binImagePlane(image::Image<PixelT> const& in, int binX, int binY) {
    int const outWidth = in.getWidth() / binX;
    int const outHeight = in.getHeight() / binY;
    auto out = std::make_shared<image::Image<PixelT>>(lsst::geom::Extent2I(outWidth, outHeight));
    out->setXY0(in.getXY0());
    std::vector<double> acc(outWidth);
    double const scale = 1.0 / (binX * binY);
    for (int oy = 0, iy = 0; oy < outHeight; ++oy) {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int i = 0; i != binY; ++i, ++iy) {
            auto acc_it = acc.begin();
            for (auto iptr = in.row_begin(iy), iend = iptr + binX * outWidth; iptr < iend;) {
                double val = static_cast<double>(*iptr);
                ++iptr;
                for (int j = 1; j != binX; ++j, ++iptr) {
                    val += static_cast<double>(*iptr);
                }
                *acc_it += val;
                ++acc_it;
            }
        }
        auto optr = out->row_begin(oy);
        for (double v : acc) {
            *optr = static_cast<PixelT>(v * scale);
            ++optr;
        }
    }
    return out;
}

// Bin a mask plane using bitwise OR so that all set flag bits are preserved.
template <typename MaskPixelT>
std::shared_ptr<image::Mask<MaskPixelT>> _binMaskPlane(image::Mask<MaskPixelT> const& in, int binX,
                                                       int binY) {
    int const outWidth = in.getWidth() / binX;
    int const outHeight = in.getHeight() / binY;
    auto out = std::make_shared<image::Mask<MaskPixelT>>(lsst::geom::Extent2I(outWidth, outHeight));
    out->setXY0(in.getXY0());
    *out = static_cast<MaskPixelT>(0);
    for (int oy = 0, iy = 0; oy < outHeight; ++oy) {
        for (int i = 0; i != binY; ++i, ++iy) {
            auto optr = out->row_begin(oy);
            for (auto iptr = in.row_begin(iy), iend = iptr + binX * outWidth; iptr < iend;) {
                MaskPixelT val = *iptr;
                ++iptr;
                for (int j = 1; j != binX; ++j, ++iptr) {
                    val |= *iptr;
                }
                *optr |= val;
                ++optr;
            }
        }
    }
    return out;
}

// Dispatch helpers called by the generic binImage template below.
template <typename PixelT>
std::shared_ptr<image::Image<PixelT>> _binImpl(image::Image<PixelT> const& in, int binX, int binY) {
    return _binImagePlane(in, binX, binY);
}

template <typename PixelT>
std::shared_ptr<image::MaskedImage<PixelT>> _binImpl(image::MaskedImage<PixelT> const& in, int binX,
                                                     int binY) {
    auto binnedImage = _binImagePlane(*in.getImage(), binX, binY);
    auto binnedVariance = _binImagePlane(*in.getVariance(), binX, binY);
    auto binnedMask = _binMaskPlane(*in.getMask(), binX, binY);
    return std::make_shared<image::MaskedImage<PixelT>>(binnedImage, binnedMask, binnedVariance);
}

}  // namespace

template <typename ImageT>
std::shared_ptr<ImageT> binImage(ImageT const& in, int const binX, int const binY,
                                 lsst::afw::math::Property const flags) {
    if (flags != lsst::afw::math::MEAN) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          (boost::format("Only afwMath::MEAN is supported, saw 0x%x") % flags).str());
    }
    if (binX <= 0 || binY <= 0) {
        throw LSST_EXCEPT(pexExcept::DomainError,
                          (boost::format("Binning must be >= 0, saw %dx%d") % binX % binY).str());
    }
    return _binImpl(in, binX, binY);
}

//
// Explicit instantiations
//
/// @cond
#define INSTANTIATE(TYPE)                                                                                  \
    template std::shared_ptr<image::Image<TYPE>> binImage(image::Image<TYPE> const&, int,                  \
                                                          lsst::afw::math::Property const);                \
    template std::shared_ptr<image::Image<TYPE>> binImage(image::Image<TYPE> const&, int, int,             \
                                                          lsst::afw::math::Property const);                \
    template std::shared_ptr<image::MaskedImage<TYPE>> binImage(image::MaskedImage<TYPE> const&, int,      \
                                                                lsst::afw::math::Property const);          \
    template std::shared_ptr<image::MaskedImage<TYPE>> binImage(image::MaskedImage<TYPE> const&, int, int, \
                                                                lsst::afw::math::Property const);

INSTANTIATE(std::uint16_t)
INSTANTIATE(int)
INSTANTIATE(float)
INSTANTIATE(double)
/// @endcond
}  // namespace math
}  // namespace afw
}  // namespace lsst
