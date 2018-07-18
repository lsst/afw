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

#include "lsst/afw/coord/Observatory.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace coord {

PYBIND11_PLUGIN(observatory) {
    py::module mod("observatory");

    py::class_<Observatory, std::shared_ptr<Observatory>> cls(mod, "Observatory");

    /* Constructors */
    cls.def(py::init<lsst::geom::Angle const, lsst::geom::Angle const, double const>());
    cls.def(py::init<std::string const, std::string const, double const>());

    /* Operators */
    cls.def("__eq__", [](Observatory const& self, Observatory const& other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](Observatory const& self, Observatory const& other) { return self != other; },
            py::is_operator());
    cls.def("__str__", &Observatory::toString);
    cls.def("__repr__", &Observatory::toString);

    /* Members */
    cls.def("getLongitude", &Observatory::getLongitude);
    cls.def("getLatitude", &Observatory::getLatitude);
    cls.def("getElevation", &Observatory::getElevation);
    cls.def("getLatitudeStr", &Observatory::getLatitudeStr);
    cls.def("getLongitudeStr", &Observatory::getLongitudeStr);
    cls.def("setLongitude", &Observatory::setLongitude, "longitude"_a);
    cls.def("setLatitude", &Observatory::setLatitude, "latitude"_a);
    cls.def("setElevation", &Observatory::setElevation, "elevation"_a);

    return mod.ptr();
}
}
}
}  // namespace lsst::afw::coord
