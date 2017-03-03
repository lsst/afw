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
//#include <pybind11/stl.h>

#include "lsst/afw/geom/ellipses/radii.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::geom::ellipses;

PYBIND11_PLUGIN(_radii) {
    py::module mod("_radii", "Python wrapper for afw _radii library");

    py::class_<DeterminantRadius> clsDeterminantRadius(mod, "DeterminantRadius");
    
    clsDeterminantRadius.def(py::init<double>(), "value"_a=1.0);
    clsDeterminantRadius.def("normalize", &DeterminantRadius::normalize);
    clsDeterminantRadius.def_static("getName", DeterminantRadius::getName);
    clsDeterminantRadius.def("__str__", [](DeterminantRadius const& self) {
        return std::to_string(self);
    });
    clsDeterminantRadius.def("__repr__", [](DeterminantRadius const& self) {
        return self.getName() + "(" + std::to_string(self) + ")";
    });

    py::class_<TraceRadius> clsTraceRadius(mod, "TraceRadius");
    
    clsTraceRadius.def(py::init<double>(), "value"_a=1.0);
    clsTraceRadius.def("normalize", &TraceRadius::normalize);
    clsTraceRadius.def_static("getName", TraceRadius::getName);
    clsTraceRadius.def("__str__", [](TraceRadius const& self) {
        return std::to_string(self);
    });
    clsTraceRadius.def("__repr__", [](TraceRadius const& self) {
        return self.getName() + "(" + std::to_string(self) + ")";
    });

    py::class_<LogDeterminantRadius> clsLogDeterminantRadius(mod, "LogDeterminantRadius");
    
    clsLogDeterminantRadius.def(py::init<double>(), "value"_a=0.0);
    clsLogDeterminantRadius.def("normalize", &LogDeterminantRadius::normalize);
    clsLogDeterminantRadius.def_static("getName", LogDeterminantRadius::getName);
    clsLogDeterminantRadius.def("__str__", [](LogDeterminantRadius const& self) {
        return std::to_string(self);
    });
    clsLogDeterminantRadius.def("__repr__", [](LogDeterminantRadius const& self) {
        return self.getName() + "(" + std::to_string(self) + ")";
    });

    py::class_<LogTraceRadius> clsLogTraceRadius(mod, "LogTraceRadius");
    
    clsLogTraceRadius.def(py::init<double>(), "value"_a=0.0);
    clsLogTraceRadius.def("normalize", &LogTraceRadius::normalize);
    clsLogTraceRadius.def_static("getName", LogTraceRadius::getName);
    clsLogTraceRadius.def("__str__", [](LogTraceRadius const& self) {
        return std::to_string(self);
    });
    clsLogTraceRadius.def("__repr__", [](LogTraceRadius const& self) {
        return self.getName() + "(" + std::to_string(self) + ")";
    });

    return mod.ptr();
}