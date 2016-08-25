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

PYBIND11_PLUGIN(_observatory) {
    py::module mod("_observatory", "Python wrapper for afw _observatory library");

    py::class_<lsst::afw::coord::Observatory> clsObservatory(mod, "Observatory");

    /* Constructors */
    clsObservatory.def(py::init<lsst::afw::geom::Angle const, lsst::afw::geom::Angle const,  double const>());
    clsObservatory.def(py::init<std::string const, std::string const, double const>());

    /* Operators */
    clsObservatory.def("__str__", [](lsst::afw::coord::Observatory &o) {
        std::ostringstream os;
        os << o;
        return os.str();
    });
    clsObservatory.def("__repr__", [](lsst::afw::coord::Observatory &o) {
        std::ostringstream os;
        os << o;
        return os.str();
    });

    /* Members */
    clsObservatory.def("getLatitude", &lsst::afw::coord::Observatory::getLatitude);
    clsObservatory.def("getLongitude", &lsst::afw::coord::Observatory::getLongitude);
    clsObservatory.def("getLatitudeStr", &lsst::afw::coord::Observatory::getLatitudeStr);
    clsObservatory.def("getLongitudeStr", &lsst::afw::coord::Observatory::getLongitudeStr);

    return mod.ptr();
}

