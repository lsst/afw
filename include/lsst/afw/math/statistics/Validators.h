// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2017 AURA/LSST.
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_MATH_STATISTICS_VALIDATORS_H
#define LSST_AFW_MATH_STATISTICS_VALIDATORS_H

#include "lsst/afw/image/MaskedImage.h"

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

/// @internal A boolean functor that always returns true
class AlwaysTrue {
public:
    template <typename... T>
    bool operator()(T...) const {
        return true;
    };
};

/// @internal A boolean functor to test |val| < limit  (for templated conditionals)
class ValidateRange {
public:
    ValidateRange(double center, double limit) : _center(center), _limit(limit) {}

    template <typename T, typename... Ts>
    bool operator()(T val, Ts...) const {
        T tmp = fabs(val - _center);
        return (tmp <= _limit);
    }

private:
    double _center;
    double _limit;
};

/// @internal A boolean functor to check for NaN or infinity (for templated conditionals)
class ValidateFinite {
public:
    template <typename T, typename... Ts>
    bool operator()(T val, Ts...) const {
        return std::isfinite(static_cast<float>(val));
    }
};

/// @internal A boolean functor to check against a mask pixel value
class ValidateMask {
public:
    explicit ValidateMask(lsst::afw::image::MaskPixel mask = 0x0) : _mask(mask) {}

    template <typename ImagePixelT, typename MaskPixelT, typename... Ts>
    bool operator()(ImagePixelT, MaskPixelT msk, Ts...) const {
        return !(msk & _mask);
    }

private:
    const lsst::afw::image::MaskPixel _mask;
};

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
