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

#include "lsst/afw/geom/CoordinateExpr.h"

namespace py = pybind11;

using namespace lsst::afw::geom;

template <int N>
void declareCoordinateExpr(py::module & mod, const std::string & suffix) {
    const std::string name = "CoordinateExpr" + suffix;
    py::class_<CoordinateExpr<N>, CoordinateBase<CoordinateExpr<N>,bool,N>> clsCoordinateExpr(mod, name.c_str());

    /* Constructors */
    clsCoordinateExpr.def(py::init<bool>(), py::arg("val")=false);

    /* Operators */
    clsCoordinateExpr.def("and_", &CoordinateExpr<N>::and_);
    clsCoordinateExpr.def("or_", &CoordinateExpr<N>::or_);
    clsCoordinateExpr.def("not_", &CoordinateExpr<N>::not_);

    mod.def("all", all<N>);
    mod.def("any", any<N>);
}

PYBIND11_PLUGIN(_coordinateExpr) {
    py::module mod("_coordinateExpr", "Python wrapper for afw _coordinateExpr library");

    declareCoordinateExpr<2>(mod, "2");
    declareCoordinateExpr<3>(mod, "3");

    return mod.ptr();
}

