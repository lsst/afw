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

#include "lsst/pex/exceptions.h"
#include "lsst/afw/math/statistics/Statistics.h"

using namespace std;
namespace pexExceptions = lsst::pex::exceptions;

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

namespace {

/// @internal end point of selection chain
/// calculate (by delegating to `calculateStatistics`) and extract selected statistics
template <typename... Ps, typename Pixel>
StatisticsResult extractStatistics(lsst::afw::image::Image<Pixel> const& img, int flags,
                                   StatisticsControl const& sctrl) {
    auto result = calculateStatistics<Ps...>(img, sctrl);

    auto output = StatisticsResult();
    if (flags & NPOINT) {
        output.count = extractOptional<Count>(result);
    }
    if (flags & MIN) {
        output.min = extractOptional<Min>(result);
    }
    if (flags & MAX) {
        output.max = extractOptional<Max>(result);
    }
    if (flags & SUM) {
        output.sum = extractOptional<Sum>(result);
    }
    if (flags & MEAN) {
        output.mean = extractOptional<Mean>(result);
        if (flags & ERRORS) {
            output.meanErr = extractOptional<MeanErr<Variance>>(result);
        }
    }
    if (flags & MEANSQUARE) {
        output.meanSquared = extractOptional<MeanSquared>(result);
        if (flags & ERRORS) {
            output.meanSquaredErr = extractOptional<MeanSquaredErr>(result);
        }
    }
    if (flags & MEDIAN) {
        if (flags & IQRANGE) {
            output.median = std::get<0>(extractOptional<Quartiles>(result));
        } else {
            output.median = extractOptional<Median>(result);
        }
    }
    if (flags & IQRANGE) {
        auto q = extractOptional<Quartiles>(result);
        output.iqrange = (std::get<2>(q) - std::get<1>(q));
    }
    if (flags & STDEV) {
        output.stddev = extractOptional<Stddev>(result);
        if (flags & ERRORS) {
            output.stddevErr = extractOptional<StddevErr>(result);
        }
    }
    if (flags & VARIANCE) {
        output.variance = extractOptional<Variance>(result);
        if (flags & ERRORS) {
            output.varianceErr = extractOptional<VarianceErr>(result);
        }
    }
    if (flags & MEANCLIP) {
        output.clippedMean = extractOptional<ClippedMean>(result);
    }
    if (flags & VARIANCECLIP) {
        output.clippedVariance = extractOptional<ClippedVariance>(result);
    }
    if (flags & STDEVCLIP) {
        output.clippedStddev = extractOptional<ClippedStddev>(result);
    }
    if (flags & ORMASK) {
        output.orMask = extractOptional<OrMask>(result);
    }
    return output;
}

/// @internal calculate errors for select products?
/// Always calculate at least Count, Mean, Variance, Stddev and OrMask.
template <typename... Ps, typename Pixel>
StatisticsResult selectErrors(lsst::afw::image::Image<Pixel> const& img, int flags,
                              StatisticsControl const& sctrl) {
    if (flags & ERRORS) {
        return extractStatistics<Ps..., Count, Mean, MeanErr<Variance>, Variance, VarianceErr, Stddev,
                                 StddevErr, OrMask>(img, flags, sctrl);
    } else {
        return extractStatistics<Ps..., Count, Mean, Variance, Stddev, OrMask>(img, flags, sctrl);
    }
}

/// @internal calculate median?
template <typename... Ps, typename Pixel>
StatisticsResult selectMedian(lsst::afw::image::Image<Pixel> const& img, int flags,
                              StatisticsControl const& sctrl) {
    if (flags & IQRANGE) {
        return selectErrors<Ps..., Quartiles>(img, flags, sctrl);
    } else if (flags & MEDIAN) {
        return selectErrors<Ps..., Median>(img, flags, sctrl);
    } else {
        return selectErrors<Ps...>(img, flags, sctrl);
    }
}

/// @internal calculate products that require clipping?
template <typename... Ps, typename Pixel>
StatisticsResult selectClipping(lsst::afw::image::Image<Pixel> const& img, int flags,
                                StatisticsControl const& sctrl) {
    if ((flags & MEANCLIP) || (flags & VARIANCECLIP) || (flags & STDEVCLIP)) {
        return selectMedian<Ps..., ClippedMean, ClippedVariance, ClippedStddev>(img, flags | IQRANGE, sctrl);
    } else {
        return selectMedian<Ps...>(img, flags, sctrl);
    }
}

/// @internal calculate min & max?
template <typename... Ps, typename Pixel>
StatisticsResult selectMinMax(lsst::afw::image::Image<Pixel> const& img, int flags,
                              StatisticsControl const& sctrl) {
    if ((flags & MIN) || (flags & MAX)) {
        return selectClipping<Ps..., Min, Max>(img, flags, sctrl);
    } else {
        return selectClipping<Ps...>(img, flags, sctrl);
    }
}

}  // namespace

/// @internal entry point to the selection chain that translates runtime into compile-time options
template <typename Pixel>
StatisticsResult standardStatistics(lsst::afw::image::Image<Pixel> const& img, int flags,
                                    StatisticsControl const& sctrl) {
    return selectMinMax(img, flags, sctrl);
}

template StatisticsResult standardStatistics(lsst::afw::image::Image<float> const& img, int,
                                             StatisticsControl const&);

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst
