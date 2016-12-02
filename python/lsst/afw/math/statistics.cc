/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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
 * see <https://www.lsstcorp.org/LegalNotices/>.
 */

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/math/Statistics.h"

namespace py = pybind11;

using namespace pybind11::literals;

using namespace lsst::afw::math;

template <typename Pixel>
void declareStatistics(py::module & mod) {
    mod.def("makeStatistics", (Statistics (*)(lsst::afw::image::Image<Pixel> const &, lsst::afw::image::Mask<lsst::afw::image::MaskPixel> const &, int const, StatisticsControl const&)) makeStatistics<Pixel>,
            "img"_a, "msk"_a, "flags"_a, "sctrl"_a=StatisticsControl());
    mod.def("makeStatistics", (Statistics (*)(lsst::afw::image::MaskedImage<Pixel> const &, int const, StatisticsControl const&)) makeStatistics<Pixel>,
            "mimg"_a, "flags"_a, "sctrl"_a=StatisticsControl());
    mod.def("makeStatistics", (Statistics (*)(lsst::afw::image::MaskedImage<Pixel> const &, lsst::afw::image::Image<WeightPixel> const &, int const, StatisticsControl const&)) makeStatistics<Pixel>,
            "mimg"_a, "weights"_a, "flags"_a, "sctrl"_a=StatisticsControl());
    mod.def("makeStatistics", (Statistics (*)(lsst::afw::image::Mask<lsst::afw::image::MaskPixel> const &, int const, StatisticsControl const&)) makeStatistics, // this is not a template, just a regular overload
            "msk"_a, "flags"_a, "sctrl"_a=StatisticsControl());
    mod.def("makeStatistics", (Statistics (*)(lsst::afw::image::Image<Pixel> const &, int const, StatisticsControl const&)) makeStatistics<Pixel>,
            "img"_a, "flags"_a, "sctrl"_a=StatisticsControl());
}

template <typename Pixel>
void declareStatisticsVectorOverloads(py::module & mod) {
    mod.def("makeStatistics", (Statistics (*)(std::vector<Pixel> const &, int const, StatisticsControl const&)) makeStatistics<Pixel>,
            "v"_a, "flags"_a, "sctrl"_a=StatisticsControl());
    mod.def("makeStatistics", (Statistics (*)(std::vector<Pixel> const &, std::vector<WeightPixel> const &, int const, StatisticsControl const&)) makeStatistics<Pixel>,
            "v"_a, "vweights"_a, "flags"_a, "sctrl"_a=StatisticsControl());
}

PYBIND11_PLUGIN(_statistics) {
    py::module mod("_statistics", "Python wrapper for afw _statistics library");

    /* Module level */
    py::enum_<Property> clsProperty(mod, "Property", py::arithmetic());
    clsProperty.value("NOTHING", Property::NOTHING)
        .value("ERRORS", Property::ERRORS)
        .value("NPOINT", Property::NPOINT)
        .value("MEAN", Property::MEAN)
        .value("STDEV", Property::STDEV)
        .value("VARIANCE", Property::VARIANCE)
        .value("MEDIAN", Property::MEDIAN)
        .value("IQRANGE", Property::IQRANGE)
        .value("MEANCLIP", Property::MEANCLIP)
        .value("STDEVCLIP", Property::STDEVCLIP)
        .value("VARIANCECLIP", Property::VARIANCECLIP)
        .value("MIN", Property::MIN)
        .value("MAX", Property::MAX)
        .value("SUM", Property::SUM)
        .value("MEANSQUARE", Property::MEANSQUARE)
        .value("ORMASK", Property::ORMASK)
        .export_values();
    // TODO: pybind11 explicit operator is not needed once DM-7974 is merged
    clsProperty.def("__or__",
        [](Property const & self, Property const & rhs) {
            return static_cast<Property>(self | rhs);
        });

    mod.def("stringToStatisticsProperty", stringToStatisticsProperty);

    py::class_<StatisticsControl, std::shared_ptr<StatisticsControl>> clsStatisticsControl(mod, "StatisticsControl");

    clsStatisticsControl.def(py::init<>());

    clsStatisticsControl.def("getMaskPropagationThreshold", &StatisticsControl::getMaskPropagationThreshold);
    clsStatisticsControl.def("setMaskPropagationThreshold", &StatisticsControl::setMaskPropagationThreshold);
    clsStatisticsControl.def("getNumSigmaClip", &StatisticsControl::getNumSigmaClip);
    clsStatisticsControl.def("getNumIter", &StatisticsControl::getNumIter);
    clsStatisticsControl.def("getAndMask", &StatisticsControl::getAndMask);
    clsStatisticsControl.def("getNoGoodPixelsMask", &StatisticsControl::getNoGoodPixelsMask);
    clsStatisticsControl.def("getNanSafe", &StatisticsControl::getNanSafe);
    clsStatisticsControl.def("getWeighted", &StatisticsControl::getWeighted);
    clsStatisticsControl.def("getWeightedIsSet", &StatisticsControl::getWeightedIsSet);
    clsStatisticsControl.def("getCalcErrorFromInputVariance", &StatisticsControl::getCalcErrorFromInputVariance);
    clsStatisticsControl.def("setNumSigmaClip", &StatisticsControl::setNumSigmaClip);
    clsStatisticsControl.def("setNumIter", &StatisticsControl::setNumIter);
    clsStatisticsControl.def("setAndMask", &StatisticsControl::setAndMask);
    clsStatisticsControl.def("setNoGoodPixelsMask", &StatisticsControl::setNoGoodPixelsMask);
    clsStatisticsControl.def("setNanSafe", &StatisticsControl::setNanSafe);
    clsStatisticsControl.def("setWeighted", &StatisticsControl::setWeighted);
    clsStatisticsControl.def("setCalcErrorFromInputVariance", &StatisticsControl::setCalcErrorFromInputVariance);

    py::class_<Statistics> clsStatistics(mod, "Statistics");

    clsStatistics.def("getResult", &Statistics::getResult,
                      "prop"_a=Property::NOTHING);
    clsStatistics.def("getError", &Statistics::getError,
                      "prop"_a=Property::NOTHING);
    clsStatistics.def("getValue", &Statistics::getValue,
                      "prop"_a=Property::NOTHING);
    clsStatistics.def("getOrMask", &Statistics::getOrMask);

    declareStatistics<unsigned short>(mod);
    declareStatistics<double>(mod);
    declareStatistics<float>(mod);
    declareStatistics<int>(mod);

    // Declare vector overloads separately to prevent casting errors
    // that otherwise (mysteriously) occur when overloads are tried
    // in order.
    declareStatisticsVectorOverloads<unsigned short>(mod);
    declareStatisticsVectorOverloads<double>(mod);
    declareStatisticsVectorOverloads<float>(mod);
    declareStatisticsVectorOverloads<int>(mod);

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}