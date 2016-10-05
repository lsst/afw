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

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/geom/LinearTransform.h"

namespace py = pybind11;

using namespace lsst::afw::geom;

PYBIND11_PLUGIN(_linearTransform) {
    py::module mod("_linearTransform", "Python wrapper for afw _linearTransform library");

    if (_import_array() < 0) {
            PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
            return nullptr;
        }

    py::class_<LinearTransform> clsLinearTransform(mod, "LinearTransform");

    /* Member types and enums */
    py::enum_<LinearTransform::Parameters>(clsLinearTransform, "Parameters")
        .value("XX", LinearTransform::Parameters::XX)
        .value("YX", LinearTransform::Parameters::YX)
        .value("XY", LinearTransform::Parameters::XY)
        .value("YY", LinearTransform::Parameters::YY)
        .export_values();

    /* Constructors */
    clsLinearTransform.def(py::init<>());
    clsLinearTransform.def(py::init<typename LinearTransform::Matrix const &>());

    /* Operators */

    /* Members */
    clsLinearTransform.def_static("makeScaling", (LinearTransform (*)(double)) LinearTransform::makeScaling);
    clsLinearTransform.def_static("makeScaling", (LinearTransform (*)(double, double)) LinearTransform::makeScaling);
    clsLinearTransform.def_static("makeRotation", (LinearTransform (*)(Angle t)) LinearTransform::makeRotation);
//    clsLinearTransform.def("getParameterVector", (ParameterVector const (LinearTransform::*)() const) &LinearTransform::getParameterVector);
    clsLinearTransform.def("getMatrix", (typename LinearTransform::Matrix const & (LinearTransform::*)() const) &LinearTransform::getMatrix);
    clsLinearTransform.def("invert", &LinearTransform::invert);
    clsLinearTransform.def("computeDeterminant", &LinearTransform::computeDeterminant);
    clsLinearTransform.def("isIdentity", &LinearTransform::isIdentity);

    return mod.ptr();
}