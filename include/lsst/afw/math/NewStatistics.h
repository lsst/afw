// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008-2017 LSST Corporation.
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

#ifndef LSST_AFW_MATH_NEWSTATISTICS_H
#define LSST_AFW_MATH_NEWSTATISTICS_H

#include <vector>
#include <iostream>

#include "ndarray.h"

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/pex/config.h"
#include "lsst/pex/exceptions.h"

namespace lsst {
namespace afw {
namespace math {

class NewStatisticsControl {
public:
    NewStatisticsControl()
            : numSigmaClip(3),
              numIter(3),
              andMask(0x0),
              noGoodPixelsMask(0x0),
              isNanSafe(true),
              calcErrorFromInputVariance(true),
              baseCaseSize(100),
              maskPropagationThresholds(16) {}

    LSST_CONTROL_FIELD(numSigmaClip, double, "Number of standard deviations to clip at");
    LSST_CONTROL_FIELD(numIter, int, "Number of iterations");
    LSST_CONTROL_FIELD(andMask, typename image::MaskPixel, "and-Mask to specify which mask planes to ignore");
    LSST_CONTROL_FIELD(noGoodPixelsMask, typename image::MaskPixel, "mask to set if no values are acceptable");
    LSST_CONTROL_FIELD(isNanSafe, bool, "Check for NaNs & Infs before running (slower)");
    LSST_CONTROL_FIELD(calcErrorFromInputVariance, bool,
                       "Calculate errors from the input variances, if available");
    LSST_CONTROL_FIELD(baseCaseSize, int, "Size of base case in partial sum for numerical stability");
    LSST_CONTROL_FIELD(
            maskPropagationThresholds, std::vector<double>,
            "Thresholds for when to propagate mask bits, treated like a dict (unset bits are set to 1.0)");
};

struct Result {
    Result();

    size_t count;
    double mean;
    double biasedVariance;
    double variance;
    double median;
    double iqrange;
    double min;
    double max;
    typename image::MaskPixel allPixelOrMask;
};

template <typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
Result standardStatistics(ImageT const &image, MaskT const *mask, WeightT const *weight,
                          VarianceT const *variance, bool computeRange, bool computeMedian, bool sigmaClipped,
                          NewStatisticsControl const &sctrl = NewStatisticsControl());

// Overloads
template <typename PixelT>
Result standardStatistics(image::Image<PixelT> const &image, image::Mask<image::MaskPixel> const &mask,
                          bool computeRange, bool computeMedian, bool sigmaClipped,
                          NewStatisticsControl const &sctrl = NewStatisticsControl()) {
    return standardStatistics(image.getArray(), mask.getArray(), nullptr, computeRange, computeMedian,
                              sigmaClipped, sctrl);
}

}  // math
}  // afw
}  // lsst

#endif
