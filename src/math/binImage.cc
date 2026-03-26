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

#include <memory>

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/offsetImage.h"

namespace pexExcept = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace math {

/*
 * Bin an Image or MaskedImage by an integral factor (the same in x and y)
 */
template <typename ImageT>
std::shared_ptr<ImageT> binImage(ImageT const& in, int const binsize, lsst::afw::math::Property const flags) {
    return binImage(in, binsize, binsize, flags);
}

/**
 * Helper class to generalize Image and MaskedImage properties without
 * directly using their Pixel templates.
 *
 * binImage can and was implemented with Pixel operators only, but rounding
 * correctly would require implementing round operator/functions, whereas the
 * implementations using primitives are easier to follow.
 *
 * @tparam T The numeric type of the image plane values.
 */
template <typename T>
struct ImagePixelType;

template <typename T>
struct ImagePixelType<image::Image<T>> {
    // Needed to resolve is_integral, is_integer, etc.
    using ImageType = T;

    using Image = image::Image<T>;
    // Placeholders are needed here, so they may as well be defaults
    using Mask = image::Image<image::MaskPixel>;
    using Variance = image::Image<image::VariancePixel>;

    using MaskType = Mask;

    // Needed for constexpr usage
    static const bool has_mask = false;
    static const bool has_variance = false;

    static Image& get_image(image::Image<T>& in) { return in; }
    static Image const& get_image_const(image::Image<T> const& in) { return in; }
    static Mask* get_mask(image::Image<T>& in) { return nullptr; }
    static Mask* const get_mask_const(image::Image<T> const& in) { return nullptr; }
    static Variance* get_variance(image::Image<T>& in) { return nullptr; }
    static Variance* const get_variance_const(image::Image<T> const& in) { return nullptr; }
};

template <typename T>
struct ImagePixelType<image::MaskedImage<T>> {
    // This is a math primitive (int, double) and not tuple/Pixel.
    using ImageType = T;
    // One could use this for most operations as the += operator is defined.
    // However, signed and unsigned accumulators would still be needed.
    using FloatAccumulator = image::pixel::SinglePixel<double, image::MaskPixel>;

    using Image = image::MaskedImage<T>::Image;
    using Mask = image::MaskedImage<T>::Mask;
    using Variance = image::MaskedImage<T>::Variance;

    using MaskType = typename image::MaskedImage<T>::SinglePixel::MaskPixelT;

    // MaskedImage enforces initialization of a mask and variance. If the
    // variance were allowed to be nullptr, this would have to change.
    static const bool has_mask = true;
    static const bool has_variance = true;

    static Image& get_image(image::MaskedImage<T>& in) { return *(in.getImage()); }
    static Image const& get_image_const(image::MaskedImage<T> const& in) { return *(in.getImage()); }
    static Mask* get_mask(image::MaskedImage<T>& in) { return in.getMask().get(); }
    static Mask* const get_mask_const(image::MaskedImage<T> const& in) { return in.getMask().get(); }
    static Variance* get_variance(image::MaskedImage<T>& in) { return in.getVariance().get(); }
    static Variance* const get_variance_const(image::MaskedImage<T> const& in) {
        return in.getVariance().get();
    }
};

template <typename ImageT>
std::shared_ptr<ImageT> binImage(ImageT const& in_ref, int const binX, int const binY,
                                 lsst::afw::math::Property const flags) {
    if (flags != lsst::afw::math::MEAN) {
        throw LSST_EXCEPT(pexExcept::InvalidParameterError,
                          (boost::format("Only afwMath::MEAN is supported, saw 0x%x") % flags).str());
    }
    if (!((binX > 0) && (binY > 0))) {
        throw LSST_EXCEPT(pexExcept::DomainError,
                          (boost::format("Binning must be > 0, saw %dx%d") % binX % binY).str());
    }

    using ImageType = typename ImagePixelType<ImageT>::ImageType;
    using ImagePlane = typename ImagePixelType<ImageT>::Image;
    using MaskPlane = typename ImagePixelType<ImageT>::Mask;
    using VariancePlane = typename ImagePixelType<ImageT>::Variance;
    using MaskType = typename ImagePixelType<ImageT>::MaskType;

    ImagePlane const& in = ImagePixelType<ImageT>::get_image_const(in_ref);
    static constexpr bool has_mask = ImagePixelType<ImageT>::has_mask;
    static constexpr bool has_variance = ImagePixelType<ImageT>::has_variance;
    // These still have to exist even for Images where they're nullptr, because
    // there is no special constexpr ternary to ignore the type of *Plane.
    VariancePlane const* const in_var =
            has_variance ? ImagePixelType<ImageT>::get_variance_const(in_ref) : nullptr;
    MaskPlane const* const in_mask = has_mask ? ImagePixelType<ImageT>::get_mask_const(in_ref) : nullptr;

    unsigned int binX_u = binX;
    unsigned int binY_u = binY;

    unsigned long long binXY_u = binX * binY;
    long long binXY_l = binX * binY;
    double binXY = double(binXY_u);

    unsigned int const outWidth = in.getWidth() / binX_u;
    unsigned int const outHeight = in.getHeight() / binY_u;

    static constexpr bool is_integer = std::is_integral<ImageType>();
    static constexpr bool is_signed = std::is_signed<ImageType>();

    // Initialize pointers to each output plane, as for the inputs
    std::shared_ptr<ImageT> out_ptr =
            std::shared_ptr<ImageT>(new ImageT(lsst::geom::Extent2I(outWidth, outHeight)));
    out_ptr->setXY0(in.getXY0());
    *out_ptr = typename ImageT::SinglePixel(0);
    auto& out_image = ImagePixelType<ImageT>::get_image(*out_ptr);
    auto* out_var = has_variance ? ImagePixelType<ImageT>::get_variance(*out_ptr) : nullptr;
    auto* out_mask = has_mask ? ImagePixelType<ImageT>::get_mask(*out_ptr) : nullptr;

    // Initialize all of the loop intermediate values
    long long remainder_l;
    unsigned long long remainder_u;
    std::vector<long long> remainders;
    std::vector<unsigned long long> remainders_u;
    if constexpr (is_integer) {
        if constexpr (is_signed) {
            remainders.resize(outWidth);
        } else {
            remainders_u.resize(outWidth);
        }
    }
    unsigned int ir;

    MaskType bitmask;
    double sum;
    long long sum_l;
    unsigned long long sum_u;

    double sum_var;

    for (unsigned int oy = 0, iy = 0; oy < outHeight; ++oy) {
        if constexpr (is_integer) {
            if constexpr (is_signed) {
                std::fill(remainders.begin(), remainders.end(), 0);
            } else {
                std::fill(remainders_u.begin(), remainders_u.end(), 0);
            }
        }
        for (unsigned int i = 0; i != binY_u; ++i, ++iy) {
            if constexpr (is_integer) {
                ir = 0;
            }
            auto optr = out_image.row_begin(oy);
            typename VariancePlane::x_iterator optr_var = has_variance ? out_var->row_begin(oy) : nullptr;
            typename MaskPlane::x_iterator optr_mask = has_mask ? out_mask->row_begin(oy) : nullptr;

            typename VariancePlane::x_iterator vptr = has_variance ? in_var->row_begin(iy) : nullptr;
            typename MaskPlane::x_iterator mptr = has_mask ? in_mask->row_begin(iy) : nullptr;
            for (typename ImagePlane::x_iterator iptr = in.row_begin(iy), iend = iptr + binX_u * outWidth;
                 iptr < iend;) {
                // Use the highest-precision supertype of the image type
                if constexpr (is_integer) {
                    if constexpr (is_signed) {
                        sum_l = 0;
                    } else {
                        sum_u = 0;
                    }
                } else {
                    sum = 0;
                }
                if constexpr (has_variance) {
                    sum_var = 0;
                }
                if constexpr (has_mask) {
                    bitmask = 0;
                }

                for (unsigned int j = 0; j != binX_u; ++j) {
                    auto value = *(iptr++);
                    if constexpr (is_integer) {
                        if constexpr (is_signed) {
                            sum_l += value;
                        } else {
                            sum_u += value;
                        }
                    } else {
                        sum += value;
                    }
                    if constexpr (has_variance) {
                        // For consistency with variance_divides which
                        // divides by the square of the divisor.
                        sum_var += *(vptr++) / binX;
                    }
                    if constexpr (has_mask) {
                        bitmask |= *(mptr++);
                    }
                }
                if constexpr (is_integer) {
                    if constexpr (is_signed) {
                        remainders[ir++] += sum_l % binXY_l;
                        *(optr++) += sum_l / binXY_l;
                    } else {
                        remainders_u[ir++] += sum_u % binXY_u;
                        *(optr++) += sum_u / binXY_u;
                    }
                } else {
                    sum /= binXY;
                    *(optr++) += sum;
                }
                if constexpr (has_variance) {
                    // For consistency with variance_divides which
                    // divides by the square of the divisor.
                    *(optr_var++) += sum_var / binY;
                }
                if constexpr (has_mask) {
                    *(optr_mask++) |= bitmask;
                }
            }
        }
        if constexpr (is_integer | has_variance) {
            auto optr = out_image.row_begin(oy);
            typename VariancePlane::x_iterator optr_var = has_variance ? out_var->row_begin(oy) : nullptr;
            for (unsigned int ox = 0; ox < outWidth; ++ox) {
                // The method used to do integer division, which is
                // round-to-zero in C++ but floor division in Python.
                // Rounding to the nearest integer seems to be the least
                // surprising option here, even if the prior (undocumented)
                // behaviour was to do C++ rounding.
                if constexpr (is_integer) {
                    ImageType shift = 0;
                    double remainder_d;
                    if constexpr (is_signed) {
                        remainder_l = remainders[ox];
                        shift = remainder_l / binXY_l;
                        remainder_d = double(remainder_l % binXY_l);
                    } else {
                        remainder_u = double(remainders_u[ox]);
                        shift = remainder_u / binXY_u;
                        remainder_d = double(remainder_u % binXY_u);
                    }
                    *(optr) += shift;
                    shift = *optr;
                    remainder_d /= binXY;
                    if (remainder_d == 0.5) {
                        remainder_d = (shift % 2) == 1;
                    } else if (remainder_d == -0.5) {
                        remainder_d = -((shift % 2) == 1);
                    } else {
                        remainder_d = round(remainder_d);
                    }
                    *(optr++) += remainder_d;
                }
                if constexpr (has_variance) {
                    // Defer taking the mean until after all sums are done
                    // The divisions by binX and binY above will prevent
                    // overflow, whereas dividing here limits roundoff error
                    // for small values (though it might do worse for
                    // large values, so hopefully users don't have var>1e20
                    *(optr_var++) /= binXY;
                }
            }
        }
    }

    return out_ptr;
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
