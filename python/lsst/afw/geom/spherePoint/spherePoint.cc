/*
 * LSST Data Management System
 * Copyright 2008-2017  AURA/LSST.
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
#include "pybind11/stl.h"

#include <memory>

#include "lsst/utils/python.h"
#include "lsst/afw/geom/Angle.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/SpherePoint.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {

using PySpherePoint = py::class_<SpherePoint, std::shared_ptr<SpherePoint>>;

PYBIND11_PLUGIN(spherePoint) {
    py::module mod("spherePoint");

    py::module::import("lsst.afw.geom.angle");
    py::module::import("lsst.afw.geom.coordinates");

    /* Module level */
    PySpherePoint cls(mod, "SpherePoint");

    /* Constructors */
    cls.def(py::init<Angle const &, Angle const &>(), "longitude"_a, "latitude"_a);
    cls.def(py::init<Point3D const &>(), "vector"_a);
    // do not wrap SpherePoint(double const lonLatRad[2]) because it is not as safe as the other constructors
    cls.def(py::init<SpherePoint const &>(), "other"_a);

    /* Operators */
    cls.def("__getitem__",
            [](SpherePoint const &self, std::ptrdiff_t i) { return self[utils::python::cppIndex(2, i)]; });
    cls.def("__eq__", &SpherePoint::operator==, py::is_operator());
    cls.def("__ne__", &SpherePoint::operator!=, py::is_operator());

    /* Members */
    cls.def("getLongitude", &SpherePoint::getLongitude);
    cls.def("getLatitude", &SpherePoint::getLatitude);
    cls.def("getVector", &SpherePoint::getVector);
    cls.def("atPole", &SpherePoint::atPole);
    cls.def("isFinite", &SpherePoint::isFinite);
    cls.def("bearingTo", &SpherePoint::bearingTo, "other"_a);
    cls.def("separation", &SpherePoint::separation, "other"_a);
    cls.def("rotated", &SpherePoint::rotated, "axis"_a, "amount"_a);
    cls.def("offset", &SpherePoint::offset, "bearing"_a, "amount"_a);
    cls.def("__str__", [](SpherePoint const &self) {
        std::ostringstream os;
        os << std::fixed << self;
        return os.str();
    });
    cls.def("__len__", [](SpherePoint const &) { return 2; });
    cls.def("__reduce__", [cls](SpherePoint const &self) {
        return py::make_tuple(cls,
                              py::make_tuple(py::cast(self.getLongitude()), py::cast(self.getLatitude())));
    });

    return mod.ptr();
}
}
}
}  // namespace lsst::afw::geom
