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

#include "lsst/afw/geom/ellipses/ReducedShear.h"
#include "lsst/afw/geom/ellipses/EllipticityBase.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::geom::ellipses;

PYBIND11_MODULE(reducedShear, mod) {
    py::class_<ReducedShear, detail::EllipticityBase> cls(mod, "ReducedShear");

    /* Constructors */
    cls.def(py::init<std::complex<double> const&>());
    cls.def(py::init<double, double>(), "e1"_a = 0.0, "e2"_a = 0.0);

    /* Members */
    //    cls.def("dAssign", (Jacobian (ReducedShear::*)(Distortion const &)) &ReducedShear::dAssign);
    //    cls.def("dAssign", (Jacobian (ReducedShear::*)(ReducedShear const &)) &ReducedShear::dAssign);
    cls.def("getAxisRatio", &ReducedShear::getAxisRatio);
    cls.def("normalize", &ReducedShear::normalize);
    cls.def("getName", &ReducedShear::getName);
    cls.def("__repr__", [](ReducedShear const& self) {
        return py::str("{}({}, {})").format(self.getName(), self.getE1(), self.getE2());
    });
}
