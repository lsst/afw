/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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
//#include <pybind11/stl.h>

#include "lsst/afw/cameraGeom/CameraPoint.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

PYBIND11_PLUGIN(_cameraPoint) {
    py::module mod("_cameraPoint", "Python wrapper for afw _cameraPoint library");

    /* Module level */
    py::class_<CameraPoint> cls(mod, "CameraPoint");

    /* Member types and enums */

    /* Constructors */
    cls.def(py::init<geom::Point2D, CameraSys const &>(), "point"_a, "cameraSys"_a);

    /* Operators */
    cls.def("__eq__", [](CameraPoint const &self, CameraPoint const &other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](CameraPoint const &self, CameraPoint const &other) { return self != other; },
            py::is_operator());
    cls.def("__str__", [](CameraPoint const &self) {
        std::ostringstream os;
        os << self;
        return os.str();
    });
    cls.def("__repr__", [](CameraPoint &self) {
        std::ostringstream os;
        os << self;
        return os.str();
    });

    /* Members */
    cls.def("getPoint", &CameraPoint::getPoint);
    cls.def("getCameraSys", &CameraPoint::getCameraSys);

    return mod.ptr();
}
}
}
}
