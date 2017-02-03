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
#include "pybind11/operators.h"

#include "lsst/afw/geom/Functor.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst { namespace afw { namespace geom { namespace {

using PyFunctor = py::class_<Functor>;
using PyLinearFunctor = py::class_<LinearFunctor,Functor>;

PYBIND11_PLUGIN(_functor) {
    py::module mod("_functor");

    /* Functor */

    PyFunctor clsFunctor(mod, "Functor");
    clsFunctor.def("__call__", &Functor::operator());
    clsFunctor.def("inverse", &Functor::inverse, "y"_a, "tol"_a=1e-10, "maxiter"_a=1000);
    clsFunctor.def("derivative", &Functor::derivative);

    /* LinearFunctor */

    PyLinearFunctor clsLinearFunctor(mod, "LinearFunctor");
    clsLinearFunctor.def(py::init<double, double>(), "slope"_a, "intercept"_a);
    clsLinearFunctor.def("__call__", &LinearFunctor::operator());
    clsLinearFunctor.def("derivative", &LinearFunctor::derivative);

    return mod.ptr();
}

}}}} // namespace lsst::afw::geom::<anonymous>
