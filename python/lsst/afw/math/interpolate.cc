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

#include "ndarray/pybind11.h"

#include "lsst/afw/math/Interpolate.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::math;

PYBIND11_MODULE(interpolate, mod) {
    py::class_<Interpolate, std::shared_ptr<Interpolate>> clsInterpolate(mod, "Interpolate");
    py::enum_<Interpolate::Style>(clsInterpolate, "Style")
            .value("UNKNOWN", Interpolate::Style::UNKNOWN)
            .value("CONSTANT", Interpolate::Style::CONSTANT)
            .value("LINEAR", Interpolate::Style::LINEAR)
            .value("NATURAL_SPLINE", Interpolate::Style::NATURAL_SPLINE)
            .value("CUBIC_SPLINE", Interpolate::Style::CUBIC_SPLINE)
            .value("CUBIC_SPLINE_PERIODIC", Interpolate::Style::CUBIC_SPLINE_PERIODIC)
            .value("AKIMA_SPLINE", Interpolate::Style::AKIMA_SPLINE)
            .value("AKIMA_SPLINE_PERIODIC", Interpolate::Style::AKIMA_SPLINE_PERIODIC)
            .value("NUM_STYLES", Interpolate::Style::NUM_STYLES)
            .export_values();

    clsInterpolate.def("interpolate", [](Interpolate &t, double const x) {
        /*
        We use a lambda function here because interpolate (with a double) is a virtual function and therefor
        cannot be wrapped directly.
        */
        return t.interpolate(x);
    });

    clsInterpolate.def("interpolate",
                       (std::vector<double> (Interpolate::*)(std::vector<double> const &) const) &
                               Interpolate::interpolate);
    clsInterpolate.def(
            "interpolate",
            (ndarray::Array<double, 1> (Interpolate::*)(ndarray::Array<double const, 1> const &) const) &
                    Interpolate::interpolate);

    mod.def("makeInterpolate",
            (std::shared_ptr<Interpolate>(*)(std::vector<double> const &, std::vector<double> const &,
                                             Interpolate::Style const))makeInterpolate,
            "x"_a, "y"_a, "style"_a = Interpolate::AKIMA_SPLINE);
    mod.def("makeInterpolate", (std::shared_ptr<Interpolate>(*)(ndarray::Array<double const, 1> const &,
                                                                ndarray::Array<double const, 1> const &y,
                                                                Interpolate::Style const))makeInterpolate,
            "x"_a, "y"_a, "style"_a = Interpolate::AKIMA_SPLINE);

    mod.def("stringToInterpStyle", stringToInterpStyle, "style"_a);
    mod.def("lookupMaxInterpStyle", lookupMaxInterpStyle, "n"_a);
    mod.def("lookupMinInterpPoints", lookupMinInterpPoints, "style"_a);
}