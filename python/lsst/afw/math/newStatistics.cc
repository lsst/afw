/*
* LSST Data Management System
* See COPYRIGHT file at the top of the source tree.
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

#include "lsst/afw/math/NewStatistics.h"
#include "lsst/pex/config/python.h"  // for LSST_DECLARE_CONTROL_FIELD

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {

template <typename ImageT, typename MaskT, typename WeightT, typename VarianceT>
void declareStandardStatistics(py::module& mod) {
    mod.def("standardStatistics", &standardStatistics<ImageT, MaskT, WeightT, VarianceT>, "image"_a, "mask"_a,
            "weight"_a, "variance"_a, "computeRange"_a, "computeMedian"_a, "sigmaClipped"_a,
            "sctrl"_a = NewStatisticsControl());
}

PYBIND11_PLUGIN(_newStatistics) {
    py::module mod("_newStatistics");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::class_<NewStatisticsControl> clsNewStatisticsControl(mod, "NewStatisticsControl");
    clsNewStatisticsControl.def(py::init<>());
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, numSigmaClip);
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, numIter);
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, andMask);
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, noGoodPixelsMask);
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, isNanSafe);
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, calcErrorFromInputVariance);
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, maskPropagationThresholds);
    LSST_DECLARE_CONTROL_FIELD(clsNewStatisticsControl, NewStatisticsControl, baseCaseSize);

    py::class_<Result> clsStandardStatisticsResult(mod, "Result");

    clsStandardStatisticsResult.def_readonly("count", &Result::count);
    clsStandardStatisticsResult.def_readonly("mean", &Result::mean);
    clsStandardStatisticsResult.def_readonly("biasedVariance", &Result::biasedVariance);
    clsStandardStatisticsResult.def_readonly("variance", &Result::variance);
    clsStandardStatisticsResult.def_readonly("min", &Result::min);
    clsStandardStatisticsResult.def_readonly("max", &Result::max);
    clsStandardStatisticsResult.def_readonly("median", &Result::median);

    declareStandardStatistics<ndarray::Array<double, 1, 1>, ndarray::Array<typename image::MaskPixel, 1, 1>,
                              ndarray::Array<double, 1, 1>, ndarray::Array<float, 1, 1>>(mod);
    declareStandardStatistics<std::vector<double>, std::vector<std::uint16_t>,
                              std::vector<double>, std::vector<float>>(mod);

    return mod.ptr();
}

}  // math
}  // afw
}  // lsst
