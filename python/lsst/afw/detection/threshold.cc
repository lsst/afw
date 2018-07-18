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
#include <pybind11/stl.h>

#include "lsst/afw/detection/Threshold.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

PYBIND11_PLUGIN(threshold) {
    py::module mod("threshold");

    py::class_<Threshold, std::shared_ptr<Threshold>> clsThreshold(mod, "Threshold");

    /* Member types and enums */
    py::enum_<Threshold::ThresholdType>(clsThreshold, "ThresholdType")
            .value("VALUE", Threshold::ThresholdType::VALUE)
            .value("BITMASK", Threshold::ThresholdType::BITMASK)
            .value("STDEV", Threshold::ThresholdType::STDEV)
            .value("VARIANCE", Threshold::ThresholdType::VARIANCE)
            .value("PIXEL_STDEV", Threshold::ThresholdType::PIXEL_STDEV)
            .export_values();

    /* Constructors */
    clsThreshold.def(
            py::init<double const, typename Threshold::ThresholdType const, bool const, double const>(),
            "value"_a, "type"_a = Threshold::VALUE, "polarity"_a = true, "includeMultiplier"_a = 1.0);

    /* Members */
    clsThreshold.def("getType", &Threshold::getType);
    clsThreshold.def_static("parseTypeString", Threshold::parseTypeString);
    clsThreshold.def_static("getTypeString", Threshold::getTypeString);
    clsThreshold.def("getValue", (double (Threshold::*)(const double) const) & Threshold::getValue,
                     "param"_a = -1);
    //
    //    template<typename ImageT>
    //    double getValue(ImageT const& image) const;
    //
    clsThreshold.def("getPolarity", &Threshold::getPolarity);
    clsThreshold.def("setPolarity", &Threshold::setPolarity);
    clsThreshold.def("getIncludeMultiplier", &Threshold::getIncludeMultiplier);
    clsThreshold.def("setIncludeMultiplier", &Threshold::setIncludeMultiplier);

    /* Module level */
    mod.def("createThreshold", createThreshold, "value"_a, "type"_a = "value", "polarity"_a = true);

    return mod.ptr();
}
}
}
}  // lsst::afw::detection