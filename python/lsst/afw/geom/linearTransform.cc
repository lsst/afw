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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/utils/python.h"

#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/LinearTransform.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

using PyLinearTransform = py::class_<LinearTransform>;

PYBIND11_PLUGIN(_linearTransform) {
    py::module mod("_linearTransform");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    PyLinearTransform clsLinearTransform(mod, "LinearTransform");

    /* Member types and enums */
    py::enum_<LinearTransform::Parameters>(clsLinearTransform, "Parameters", py::arithmetic())
        .value("XX", LinearTransform::Parameters::XX)
        .value("YX", LinearTransform::Parameters::YX)
        .value("XY", LinearTransform::Parameters::XY)
        .value("YY", LinearTransform::Parameters::YY)
        .export_values();

    /* Constructors */
    clsLinearTransform.def(py::init<>());
    clsLinearTransform.def(py::init<typename LinearTransform::Matrix const &>(), "matrix"_a);

    /* Operators */
    clsLinearTransform.def("__call__",
                           (Point2D(LinearTransform::*)(Point2D const &) const) & LinearTransform::operator(),
                           py::is_operator());
    clsLinearTransform.def(
        "__call__", (Extent2D(LinearTransform::*)(Extent2D const &) const) & LinearTransform::operator(),
        py::is_operator());
    clsLinearTransform.def(
        "__getitem__", [](LinearTransform const &self, int i) { return self[lsst::utils::python::cppIndex(4, i)]; },
        py::is_operator());
    clsLinearTransform.def("__getitem__",
                           [](LinearTransform const &self, std::pair<int, int> i) {
                               auto row = lsst::utils::python::cppIndex(2, i.first);
                               auto col = lsst::utils::python::cppIndex(2, i.second);
                               return self.getMatrix()(row, col);
                           },
                           py::is_operator());
    clsLinearTransform.def("__mul__", &LinearTransform::operator*, py::is_operator());
    clsLinearTransform.def("__add__", &LinearTransform::operator+, py::is_operator());
    clsLinearTransform.def("__sub__", &LinearTransform::operator-, py::is_operator());
    clsLinearTransform.def("__iadd__", &LinearTransform::operator+=, py::is_operator());
    clsLinearTransform.def("__isub__", &LinearTransform::operator-=, py::is_operator());

    /* Members */
    clsLinearTransform.def_static("makeScaling", (LinearTransform(*)(double))LinearTransform::makeScaling,
                                  "scale"_a);
    clsLinearTransform.def_static("makeScaling",
                                  (LinearTransform(*)(double, double))LinearTransform::makeScaling);
    clsLinearTransform.def_static("makeRotation", (LinearTransform(*)(Angle t))LinearTransform::makeRotation,
                                  "angle"_a);
    clsLinearTransform.def("getParameterVector", &LinearTransform::getParameterVector);
    clsLinearTransform.def(
        "getMatrix",
        (typename LinearTransform::Matrix const &(LinearTransform::*)() const) & LinearTransform::getMatrix);
    clsLinearTransform.def("invert", &LinearTransform::invert);
    clsLinearTransform.def("computeDeterminant", &LinearTransform::computeDeterminant);
    clsLinearTransform.def("isIdentity", &LinearTransform::isIdentity);

    clsLinearTransform.def("set", [](LinearTransform &self, double xx, double yx, double xy, double yy) {
        self[LinearTransform::XX] = xx;
        self[LinearTransform::XY] = xy;
        self[LinearTransform::YX] = yx;
        self[LinearTransform::YY] = yy;
    }, "xx"_a, "yx"_a, "xy"_a, "yy"_a);

    return mod.ptr();
}

}}}}  // namespace lsst::afw::geom::<anonymous>
