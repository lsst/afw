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

#ifndef LSST_AFW_MATH_STATISTICS_MULTIPASSSTATISTICS_H
#define LSST_AFW_MATH_STATISTICS_MULTIPASSSTATISTICS_H

#include "ndarray.h"

#include "lsst/pex/config.h"
#include "lsst/pex/exceptions.h"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/statistics/Framework.h"
#include "lsst/afw/math/statistics/Products.h"
#include "lsst/afw/math/statistics/Validators.h"
#include "lsst/afw/math/statistics/SinglePassStatistics.h"

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

/**
 * Functor to calculate statistics that (may) require multiple passes over the data (e.g. `ClippedMean`).
 *
 * This delegates to `SinglePassStatistics` for each pass.
 * An initial pass calculates all requested statistics except for those that require clipping.
 * Then (if needed) a number of clipped passes are performed, calculating only those statistics that require
 * it.
 *
 * @tparam Product a `ConcreteProduct` containing the statistics products to calculate.
 * @tparam Validator an optional validator type to be given to `SinglePassStatistics`.
 *         Note that for the clipped pass the input validator is combined with a `ValidateRange` instance.
 */
template <typename Product, typename Validator = AlwaysTrue>
class MultiPassStatistics {
public:
    /**
     * Default constructor
     */
    MultiPassStatistics() : _validator() {}

    /**
     * Constructor with user provided validator
     *
     * @param validator user provided Validator instance (e.g. `ValidateFinite`)
     *
     * @note that the validator is copied
     */
    explicit MultiPassStatistics(Validator const& validator) : _validator(validator) {}

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
        SinglePassStatistics<Product, Validator> initialPass(_validator);

        auto result = initialPass(img, msk, var, wgt, sctrl);

        // (optionally) run clipped pass
        return clippedPass(result, img, msk, var, wgt, sctrl);
    }

private:
    /**
     * Calculate clipped statistics (if requested)
     *
     * This member function is disabled (using SFINAE) if none of the requested products require clipping
     *
     * @param result the result of an initial (non-clipped) run of `SinglePassStatistics(...)`
     * @param img, Image (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param msk, Mask (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param var, Variance (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param wgt, Weights (assumed to implement interface equivalent to that of `ndarray::Array<T, 2, 1>`)
     * @param sctrl Control how things are calculated
     *
     * @return result updated with clipped statistics
     *
     * @note, currently only `ClippedMean`, `ClippedVariance` and `ClippedStddev` are supported
     */
    template <typename P, typename ImageT, typename MaskT, typename WeightT, typename VarianceT,
              typename std::enable_if<
                      IsInProductSet<ClippedMean, typename P::Intermediate>::value &&
                              IsInProductSet<ClippedVariance, typename P::Intermediate>::value &&
                              IsInProductSet<Quartiles, typename P::Intermediate>::value,
                      int>::type = 0>
    Product clippedPass(P& result, ImageT const& img, MaskT const& msk, VarianceT const& var,
                        WeightT const& wgt, StatisticsControl const& sctrl) {
        // Calculate clipping range for first clipped pass
        auto quartiles = extract<Quartiles>(result);
        auto center = std::get<0>(quartiles);
        auto limit = sctrl.numSigmaClip * IQ_TO_STDEV * (std::get<2>(quartiles) - std::get<1>(quartiles));

        ConcreteProduct<Mean, Variance> clippedResult;
        for (int i = 0; i < sctrl.numIter; ++i) {
            // Create a new validator to do the clipping
            // (an alternative would be to update an existing validator,
            // which may or may not be more efficient depending on the optimizer)
            auto validator = makeCombinedValidator(_validator, ValidateRange(center, limit));

            clippedResult = SinglePassStatistics<decltype(clippedResult), decltype(validator)>(validator)(
                    img, msk, var, wgt, sctrl);

            // Update clipping range
            center = extract<Mean>(clippedResult);
            limit = sctrl.numSigmaClip * std::sqrt(extract<Variance>(clippedResult));
        }

        // Merge `clippedResult` values into `result` product
        result.template get<ClippedMean>() = clippedResult;
        result.template get<ClippedVariance>() = clippedResult;

        return result;
    }

    /**
     * SFINAE construct to disable clipping when not needed
     */
    template <typename P, typename ImageT, typename MaskT, typename WeightT, typename VarianceT,
              typename std::enable_if<
                      !IsInProductSet<ClippedMean, typename P::Intermediate>::value ||
                              !IsInProductSet<ClippedVariance, typename P::Intermediate>::value ||
                              !IsInProductSet<Quartiles, typename P::Intermediate>::value,
                      int>::type = 0>
    Product clippedPass(P& initialPass, ImageT const& img, MaskT const& msk, VarianceT const& var,
                        WeightT const& wgt, StatisticsControl const& sctrl) {
        // noop
        return initialPass;
    }

    /// Used to validate input values
    const Validator _validator;
};

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
