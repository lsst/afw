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

#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::geom;
using namespace ellipses;

PYBIND11_PLUGIN(_ellipse) {
    py::module mod("_ellipse", "Python wrapper for afw _ellipse library");

    /* Module level */
    py::class_<Ellipse, std::shared_ptr<Ellipse>> clsEllipse(mod, "Ellipse");

    /* Member types and enums */

    /* Constructors */
    clsEllipse.def(py::init<BaseCore const &, Point2D const &>(), "core"_a, "center"_a=Point2D());

    /* Operators */

    /* Members */
    clsEllipse.def("getCore", [](Ellipse & ellipse){
        return ellipse.getCorePtr();
    });
    clsEllipse.def("getCenter", (Point2D & (Ellipse::*)())&Ellipse::getCenter);

    return mod.ptr();
}