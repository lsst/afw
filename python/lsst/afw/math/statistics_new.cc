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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/math/statistics/Statistics.h"
#include "lsst/pex/config/python.h"  // for LSST_DECLARE_CONTROL_FIELD

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {
namespace statistics {

template <typename Pixel>
void declareStandardStatistics(py::module& mod) {
    /**
     * standardStatistics is wrapped in a lambda to translate the C++ interface (which uses enum flags) to
     * the Python interface (which takes keyword arguments).
     * If one of these interfaces is declared not needed we may change this.
     * Or perhaps a Python level wrapper is better.
     */
    mod.def("standardStatistics",
            [](lsst::afw::image::Image<Pixel> const& img, bool count, bool min, bool max, bool sum, bool mean,
               bool meanSquared, bool median, bool iqrange, bool stddev, bool variance, bool clippedMean,
               bool clippedVariance, bool clippedStddev, bool orMask, bool errors,
               StatisticsControl const& sctrl) -> py::dict {
                int flags = StatisticsProperty::NOTHING;
                if (count) {
                    flags |= NPOINT;
                }
                if (min) {
                    flags |= MIN;
                }
                if (max) {
                    flags |= MAX;
                }
                if (sum) {
                    flags |= SUM;
                }
                if (mean) {
                    flags |= MEAN;
                }
                if (meanSquared) {
                    flags |= MEANSQUARE;
                }
                if (median) {
                    flags |= MEDIAN;
                }
                if (iqrange) {
                    flags |= IQRANGE;
                }
                if (stddev) {
                    flags |= STDEV;
                }
                if (variance) {
                    flags |= VARIANCE;
                }
                if (clippedMean) {
                    flags |= MEANCLIP;
                }
                if (clippedVariance) {
                    flags |= VARIANCECLIP;
                }
                if (clippedStddev) {
                    flags |= STDEVCLIP;
                }
                if (orMask) {
                    flags |= ORMASK;
                }
                if (errors) {
                    flags |= ERRORS;
                }

                auto result = standardStatistics(img, flags, sctrl);

                py::dict output;
                if (count) {
                    output["count"] = result.count;
                }
                if (min) {
                    output["min"] = result.min;
                }
                if (max) {
                    output["max"] = result.max;
                }
                if (sum) {
                    output["sum"] = result.sum;
                }
                if (mean) {
                    output["mean"] = result.mean;
                    if (errors) {
                        output["meanErr"] = result.meanErr;
                    }
                }
                if (meanSquared) {
                    output["meanSquared"] = result.meanSquared;
                    if (errors) {
                        output["meanSquaredErr"] = result.meanSquaredErr;
                    }
                }
                if (median) {
                    output["median"] = result.median;
                }
                if (iqrange) {
                    output["iqrange"] = result.iqrange;
                }
                if (stddev) {
                    output["stddev"] = result.stddev;
                    if (errors) {
                        output["stddevErr"] = result.stddevErr;
                    }
                }
                if (variance) {
                    output["variance"] = result.variance;
                    if (errors) {
                        output["varianceErr"] = result.varianceErr;
                    }
                }
                if (clippedMean) {
                    output["clippedMean"] = result.clippedMean;
                }
                if (clippedVariance) {
                    output["clippedVariance"] = result.clippedVariance;
                }
                if (clippedStddev) {
                    output["clippedStddev"] = result.clippedStddev;
                }
                if (orMask) {
                    output["orMask"] = result.orMask;
                }
                return output;
            },
            "img"_a, "count"_a = false, "min"_a = false, "max"_a = false, "sum"_a = false, "mean"_a = false,
            "meanSquared"_a = false, "median"_a = false, "iqrange"_a = false, "stddev"_a = false,
            "variance"_a = false, "clippedMean"_a = false, "clippedVariance"_a = false,
            "clippedStddev"_a = false, "orMask"_a = false, "errors"_a = false,
            "sctrl"_a = StatisticsControl());
}

PYBIND11_PLUGIN(_statistics_new) {
    py::module mod("_statistics_new");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<StatisticsControl> clsStatisticsControl(mod, "NewStatisticsControl");
    clsStatisticsControl.def(py::init<>());
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, numSigmaClip);
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, numIter);
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, andMask);
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, noGoodPixelsMask);
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, isNanSafe);
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, calcErrorFromInputVariance);
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, maskPropagationThresholds);
    LSST_DECLARE_CONTROL_FIELD(clsStatisticsControl, StatisticsControl, baseCaseSize);

    declareStandardStatistics<float>(mod);

    return mod.ptr();
}

}  // namespace statistics
}  // namespace math
}  // namespace afw
}  // namespace lsst
