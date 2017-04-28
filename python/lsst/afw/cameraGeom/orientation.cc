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
//#include <pybind11/operators.h>
//#include <pybind11/stl.h>

#include "lsst/afw/cameraGeom/Orientation.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

PYBIND11_PLUGIN(_orientation) {
    py::module mod("_orientation", "Python wrapper for afw _orientation library");

    /* Module level */
    py::class_<Orientation> cls(mod, "Orientation");

    /* Member types and enums */

    /* Constructors */
    cls.def(py::init<geom::Point2D, geom::Point2D, geom::Angle, geom::Angle, geom::Angle>(),
            "fpPosition"_a = geom::Point2D(0, 0), "refPoint"_a = geom::Point2D(-0.5, -0.5),
            "yaw"_a = geom::Angle(0), "pitch"_a = geom::Angle(0), "roll"_a = geom::Angle(0));

    /* Operators */

    /* Members */
    cls.def("getFpPosition", &Orientation::getFpPosition);
    cls.def("getReferencePoint", &Orientation::getReferencePoint);
    cls.def("getYaw", &Orientation::getYaw);
    cls.def("getPitch", &Orientation::getPitch);
    cls.def("getRoll", &Orientation::getRoll);
    cls.def("getNQuarter", &Orientation::getNQuarter);
    cls.def("makePixelFpTransform", &Orientation::makePixelFpTransform, "pixelSizeMm"_a);
    cls.def("makeFpPixelTransform", &Orientation::makeFpPixelTransform, "pixelSizeMm"_a);
    cls.def("getFpPosition", &Orientation::getFpPosition);
    cls.def("getFpPosition", &Orientation::getFpPosition);
    cls.def("getFpPosition", &Orientation::getFpPosition);
    cls.def("getFpPosition", &Orientation::getFpPosition);

    return mod.ptr();
}
}
}
}
