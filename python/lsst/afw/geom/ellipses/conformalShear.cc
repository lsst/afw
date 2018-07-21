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

#include "lsst/afw/geom/ellipses/ConformalShear.h"
#include "lsst/afw/geom/ellipses/EllipticityBase.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::geom::ellipses;

PYBIND11_MODULE(conformalShear, mod) {
    py::class_<ConformalShear, detail::EllipticityBase> cls(mod, "ConformalShear");

    /* Constructors */
    cls.def(py::init<std::complex<double> const&>());
    cls.def(py::init<double, double>(), "e1"_a = 0.0, "e2"_a = 0.0);

    /* Members */
    //    cls.def("dAssign", (Jacobian (ConformalShear::*)(Distortion const &)) &ConformalShear::dAssign);
    //    cls.def("dAssign", (Jacobian (ConformalShear::*)(ReducedShear const &)) &ConformalShear::dAssign);
    cls.def("getAxisRatio", &ConformalShear::getAxisRatio);
    cls.def("normalize", &ConformalShear::normalize);
    cls.def("getName", &ConformalShear::getName);
    cls.def("__repr__", [](ConformalShear const& self) {
        return py::str("%s(%g, %g)").format(self.getName(), self.getE1(), self.getE2());
    });
}