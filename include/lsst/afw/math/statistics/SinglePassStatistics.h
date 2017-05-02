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

#ifndef LSST_AFW_MATH_STATISTICS_SINGLEPASSSTATISTICS_H
#define LSST_AFW_MATH_STATISTICS_SINGLEPASSSTATISTICS_H

#include <vector>

#include "ndarray.h"

#include "lsst/pex/config.h"
#include "lsst/pex/exceptions.h"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/statistics/Framework.h"
#include "lsst/afw/math/statistics/Products.h"

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

/**
 * control how things are calculated
 */
class StatisticsControl {
public:
    StatisticsControl()
            : numSigmaClip(3),
              numIter(3),
              andMask(0x0),
              noGoodPixelsMask(0x0),
              isNanSafe(true),
              calcErrorFromInputVariance(true),
              baseCaseSize(1024),
              maskPropagationThresholds(16) {}

    LSST_CONTROL_FIELD(numSigmaClip, double, "Number of standard deviations to clip at");
    LSST_CONTROL_FIELD(numIter, int, "Number of iterations");
    LSST_CONTROL_FIELD(andMask, typename image::MaskPixel, "and-Mask to specify which mask planes to ignore");
    LSST_CONTROL_FIELD(noGoodPixelsMask, typename image::MaskPixel,
                       "mask to set if no values are acceptable");
    LSST_CONTROL_FIELD(isNanSafe, bool, "Validate for NaNs & Infs before running (slower)");
    LSST_CONTROL_FIELD(calcErrorFromInputVariance, bool,
                       "Calculate errors from the input variances, if available");
    LSST_CONTROL_FIELD(baseCaseSize, int, "Size of base case in partial sum for numerical stability");
    LSST_CONTROL_FIELD(
            maskPropagationThresholds, std::vector<double>,
            "Thresholds for when to propagate mask bits, treated like a dict (unset bits are set to 1.0)");
};

/**
 * Functor to calculate statistics that require only a single pass over the data (e.g. `Mean`).
 *
 * @tparam Product a `ConcreteProduct` containing the statistics products to calculate.
 * @tparam Validator an optional validator type to be given to `SinglePassStatistics`.
 *         Note that for the clipped pass the input validator is combined with a `ValidateRange` instance.
 *
 * @note currently requires a 2d-ndarray like interface (because `ndarray::Array<T, 2, 1>` (and by
 * extension `afw::image::Image`) doesn't allow for flat iteration over the (pixel) values.
 * This may need to be changed, especially to allow for pairwise summation for increased numerical stability
 * (which otherwise is supported by the statistics framework).
 */
template <typename Product, typename Validator = AlwaysTrue>
class SinglePassStatistics {
public:
    /**
     * Default constructor
     */
    SinglePassStatistics() : _validator() {}

    /**
     * Constructor with user provided validator
     *
     * @param validator user provided Validator instance (e.g. `ValidateFinite`)
     *
     * @note that the validator is copied
     */
    explicit SinglePassStatistics(Validator const& validator) : _validator(validator) {}

    /**
     * Function call operator, calculate statistics for data provided
     *
     * @param img, Image (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param msk, Mask (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param var, Variance (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param wgt, Weights (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param sctrl Control how things are calculated
     */
    template <typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
    Product operator()(ImageT const& img, MaskT const& msk, VarianceT const& var, WeightT const& wgt,
                       StatisticsControl const& sctrl) {
        Product product;

        auto m = msk.begin();
        auto v = var.begin();
        auto w = wgt.begin();
        for (auto i = img.begin(); i != img.end(); ++i) {
            for (auto j = i->begin(); j != i->end(); ++j) {
                if (_validator(*j, *m, *v, *w)) {
                    product.accumulate(*j, *m, *v, *w);
                } else {
                    product.accumulateClipped(*j, *m, *v, *w);
                }
            }
        }
        return product;
    }

private:
    /// Used to validate input values
    const Validator _validator;
};

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
