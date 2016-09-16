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
#include <pybind11/operators.h>

#include "lsst/afw/geom/Functor.h"

namespace py = pybind11;

using namespace lsst::afw::geom;

PYBIND11_PLUGIN(_functor) {
    py::module mod("_functor", "Python wrapper for afw _functor library");

    py::class_<Functor> clsFunctor(mod, "Functor");

    /* Operators */
    clsFunctor.def("__call__", &Functor::operator());

    /* Members */
    clsFunctor.def("inverse", &Functor::inverse,
        py::arg("y"), py::arg("tol")=1e-10, py::arg("maxiter")=1000);
    clsFunctor.def("derivative", &Functor::derivative);

    py::class_<LinearFunctor> clsLinearFunctor(mod, "LinearFunctor", py::base<Functor>());

    /* Constructors */
    clsLinearFunctor.def(py::init<double, double>());

    /* Operators */
    clsLinearFunctor.def("__call__", &LinearFunctor::operator());

    /* Members */
    clsLinearFunctor.def("derivative", &LinearFunctor::derivative);

    return mod.ptr();
}

