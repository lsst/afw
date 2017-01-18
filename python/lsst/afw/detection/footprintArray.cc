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
//#include <pybind11/stl.h>

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintArray.cc"     // FootprintArray.h does not define the templates

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace detection {

namespace {
    template <typename T>
    void declareTemplates(py::module & mod) {
        namespace afwGeom = lsst::afw::geom;

        mod.def("flattenArray", (void (*)(Footprint const &,
                                          ndarray::Array<T const, 2, 0> const &,
                                          ndarray::Array<T, 1, 0> const &,
                                          afwGeom::Point2I const &)) &flattenArray<T const, T, 2, 0, 0>,
                "fp"_a, "src"_a, "dest"_a, "xy0"_a=afwGeom::Point2I());
        mod.def("flattenArray", (void (*)(Footprint const &,
                                          ndarray::Array<T const, 3, 0> const &,
                                          ndarray::Array<T, 2, 0> const &,
                                          afwGeom::Point2I const &)) &flattenArray<T const, T, 3, 0, 0>,
                "fp"_a, "src"_a, "dest"_a, "xy0"_a=afwGeom::Point2I());
        mod.def("expandArray", (void (*)(Footprint const &,
                                          ndarray::Array<T const, 1, 0> const &,
                                          ndarray::Array<T, 2, 0> const &,
                                          afwGeom::Point2I const &)) &expandArray<T const, T, 1, 0, 0>,
                "fp"_a, "src"_a, "dest"_a, "xy0"_a=afwGeom::Point2I());
        mod.def("expandArray", (void (*)(Footprint const &,
                                          ndarray::Array<T const, 2, 0> const &,
                                          ndarray::Array<T, 3, 0> const &,
                                          afwGeom::Point2I const &)) &expandArray<T const, T, 2, 0, 0>,
                "fp"_a, "src"_a, "dest"_a, "xy0"_a=afwGeom::Point2I());
    }
}

PYBIND11_PLUGIN(_footprintArray) {
    py::module mod("_footprintArray", "Python wrapper for afw _footprintArray library");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    /* Module level */
    declareTemplates<std::uint16_t>(mod);
    declareTemplates<int>(mod);
    declareTemplates<float>(mod);
    declareTemplates<double>(mod);

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}

}}}     // lsst::afw::detection
