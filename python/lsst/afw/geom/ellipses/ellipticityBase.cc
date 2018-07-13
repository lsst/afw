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

#include "lsst/afw/geom/ellipses/EllipticityBase.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::geom::ellipses;

PYBIND11_PLUGIN(ellipticityBase) {
    py::module mod("ellipticityBase");

    py::class_<detail::EllipticityBase> cls(mod, "EllipticityBase");

    /* Member types and enums */
    py::enum_<detail::EllipticityBase::ParameterEnum>(cls, "ParameterEnum")
            .value("E1", detail::EllipticityBase::ParameterEnum::E1)
            .value("E2", detail::EllipticityBase::ParameterEnum::E2)
            .export_values();

    /* Members */
    cls.def("getComplex",
            (std::complex<double> & (detail::EllipticityBase::*)()) & detail::EllipticityBase::getComplex);
    cls.def("setComplex", &detail::EllipticityBase::setComplex);
    cls.def("getE1", &detail::EllipticityBase::getE1);
    cls.def("setE1", &detail::EllipticityBase::setE1);
    cls.def("getE2", &detail::EllipticityBase::getE2);
    cls.def("setE2", &detail::EllipticityBase::setE2);
    cls.def("getTheta", &detail::EllipticityBase::getTheta);
    cls.def("__str__", [](detail::EllipticityBase const& self) {
        return py::str("(%g, %g)").format(self.getE1(), self.getE2());
    });

    return mod.ptr();
}