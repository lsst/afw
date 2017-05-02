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

#ifndef LSST_AFW_MATH_STATISTICS_CALCULATESTATISTICS_H
#define LSST_AFW_MATH_STATISTICS_CALCULATESTATISTICS_H

#include "ndarray.h"

#include "lsst/pex/config.h"
#include "lsst/pex/exceptions.h"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/statistics/MultiPassStatistics.h"

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

/// @internal substitute for Mask input and do nothing
class NoMask {
public:
    NoMask const& begin() const { return *this; }
    NoMask& begin() { return *this; }

    NoMask& operator++() { return *this; }
    NoMask& operator++(int) { return *this; }
    NoMask& operator+=(int) { return *this; }

    image::MaskPixel operator*() const { return 0x0; }
};

NoMask& operator+(NoMask& self, size_t n) { return self += n; }

/// @internal substitute for Variance input and do nothing
class NoVariance {
public:
    NoVariance const& begin() const { return *this; }
    NoVariance& begin() { return *this; }

    NoVariance& operator++() { return *this; }
    NoVariance& operator++(int) { return *this; }
    NoVariance& operator+=(int) { return *this; }

    double operator*() const { return 0.0; }
};

NoVariance& operator+(NoVariance& self, size_t n) { return self += n; }

/// @internal substitute for Weight input and do nothing
class NoWeight {
public:
    NoWeight const& begin() const { return *this; }
    NoWeight& begin() { return *this; }

    NoWeight& operator++() { return *this; }
    NoWeight& operator++(int) { return *this; }
    NoWeight& operator+=(int) { return *this; }

    double operator*() const { return 1.0; }
};

NoWeight& operator+(NoWeight& self, size_t n) { return self += n; }

/**
 * Calculate requested statistics
 *
 * Main C++ interface to calculate statistics (should be used instead of `SinglePassStatistics` or
 * `MultiPassStatistics` directly).
 *
 * Interdependent (intermediate) products are only calculated once and in a single pass
 * through the data (unless multiple passes are requested for clipping).
 * Output results only contain the requested values which can be extracted with either:
 * - `extract` (presence is checked at compile time) or
 * - `extractOptional` (presence is checked at runtime).
 *
 * For example:
 *
 *     StatisticsControl sctrl;
 *     sctrl.numIter = 5;
 *
 *     auto result = calculateStatistics<Mean, Variance, Min>(image, // e.g. ndarray::Array<float, 2, 1>
 *                                                            mask, variance, weight, sctrl);
 *     double min = extract<Min>(result);
 *     // double max = extract<Max>(result); // will not compile
 *     // double median = extractOptional<Median>(result); // will throw
 *
 * @tparam Ts parameter pack with statistics `Product`s to calculate.
 *
 * @param img, Image (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
 * @param msk, Mask (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
 * @param var, Variance (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
 * @param wgt, Weights (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
 * @param sctrl Control how things are calculated
 */
template <typename... Ts, typename ImageT, typename MaskT, typename VarianceT, typename WeightT>
ConcreteProduct<Ts...> calculateStatistics(ImageT const& img, MaskT const& msk, VarianceT const& var,
                                           WeightT const& wgt,
                                           StatisticsControl const& sctrl = StatisticsControl()) {
    if (sctrl.isNanSafe) {
        auto validator = makeCombinedValidator(ValidateMask(sctrl.andMask), ValidateFinite());
        return MultiPassStatistics<ConcreteProduct<Ts...>, decltype(validator)>(validator)(img, msk, var, wgt,
                                                                                           sctrl);
    } else {
        auto validator = ValidateMask(sctrl.andMask);
        return MultiPassStatistics<ConcreteProduct<Ts...>, decltype(validator)>(validator)(img, msk, var, wgt,
                                                                                           sctrl);
    }
}

/**
 * The calculateStatistics() overload to handle regular (non-masked) Images
 *
 * @param img Image whose properties we want
 * @param sctrl Control how things are calculated
 */
template <typename... Ts, typename Pixel>
ConcreteProduct<Ts...> calculateStatistics(lsst::afw::image::Image<Pixel> const& img,
                                           StatisticsControl const& sctrl = StatisticsControl()) {
    return calculateStatistics<Ts...>(img.getArray(), NoMask(), NoVariance(), NoWeight(), sctrl);
}

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
