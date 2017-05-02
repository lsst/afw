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

#ifndef LSST_AFW_MATH_STATISTICS_STATISTICS_H
#define LSST_AFW_MATH_STATISTICS_STATISTICS_H

#include <vector>
#include <iostream>

#include "ndarray.h"

#include "lsst/pex/config.h"
#include "lsst/pex/exceptions.h"

#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/statistics/Products.h"
#include "lsst/afw/math/statistics/calculateStatistics.h"

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

typedef lsst::afw::image::VariancePixel WeightPixel;  // Type used for weights

/**
 * control what is calculated
 */
enum StatisticsProperty {
    NOTHING = 0x0,         ///< We don't want anything
    ERRORS = 0x1,          ///< Include errors of requested quantities
    NPOINT = 0x2,          ///< number of sample points
    MEAN = 0x4,            ///< estimate sample mean
    STDEV = 0x8,           ///< estimate sample standard deviation
    VARIANCE = 0x10,       ///< estimate sample variance
    MEDIAN = 0x20,         ///< estimate sample median
    IQRANGE = 0x40,        ///< estimate sample inter-quartile range
    MEANCLIP = 0x80,       ///< estimate sample N-sigma clipped mean (N set in StatisticsControl, default=3)
    STDEVCLIP = 0x100,     ///< estimate sample N-sigma clipped stdev (N set in StatisticsControl, default=3)
    VARIANCECLIP = 0x200,  ///< estimate sample N-sigma clipped variance
                           ///<  (N set in StatisticsControl, default=3)
    MIN = 0x400,           ///< estimate sample minimum
    MAX = 0x800,           ///< estimate sample maximum
    SUM = 0x1000,          ///< find sum of pixels in the image
    MEANSQUARE = 0x2000,   ///< find mean value of square of pixel values
    ORMASK = 0x4000        ///< get the or-mask of all pixels used.
};

/**
 * Output for `standardStatistics`
 *
 * Each entry corresponds directly (except for `errors` which
 * are listed separately) to a flag in `StatisticsProperty`.
 *
 * @note that, at present, there is no shielding the user from reading results values
 * that have not been calculated. This is (not yet) done because it is unknown if this
 * interface will be needed at all at the C++ level.
 */
struct StatisticsResult {
    size_t count;
    double min;
    double max;
    double sum;
    double mean;
    double meanErr;
    double meanSquared;
    double meanSquaredErr;
    double median;
    double medianErr;
    double iqrange;
    double stddev;
    double stddevErr;
    double variance;
    double varianceErr;
    double clippedMean;
    double clippedVariance;
    double clippedStddev;
    image::MaskPixel orMask;
};

/**
 * Calculates a set of standard statistics
 *
 * This wraps the more flexible templated `calculateStatistics` function
 * by grouping a few statistics together (i.e. when you ask for min, you get max as well).
 * This is a compromise between calculating only what you need (which `calculateStatistics` provides)
 * and allowing runtime selection of options while minimizing the number of template instantiations.
 *
 * Use this interface if you need to decide which statistics to calculate at runtime,
 * use `calculateStatistics` instead if you know at compile-time what you need, or require more
 * flexibility then `standardStatistics` provides.
 *
 * @param img Image whose properties we want
 * @param flags Describes what we want to calculate
 * @param sctrl Control how things are calculated
 *
 * @returns statistics that were calculated.
 */
template <typename Pixel>
StatisticsResult standardStatistics(lsst::afw::image::Image<Pixel> const& img, int flags,
                                    statistics::StatisticsControl const& sctrl);

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst

#endif
