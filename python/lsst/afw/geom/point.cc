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
#include <pybind11/operators.h>

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace py = pybind11;

using namespace lsst::afw::geom;

template <typename T, int N>
py::class_<PointBase<T,N>> declarePointBase(py::module &mod, const std::string & suffix) {
    const std::string name = "PointBase" + suffix;
    py::class_<PointBase<T,N>> cls(mod, name.c_str(), py::base<CoordinateBase<Point<T,N>,T,N>>());

    /* Operators */
    cls.def("eq", [](PointBase<T,N> &p, Point<T,N> value) { return p.eq(value); });
    cls.def("ne", [](PointBase<T,N> &p, Point<T,N> value) { return p.ne(value); });
    cls.def("lt", [](PointBase<T,N> &p, Point<T,N> value) { return p.lt(value); });
    cls.def("le", [](PointBase<T,N> &p, Point<T,N> value) { return p.le(value); });
    cls.def("gt", [](PointBase<T,N> &p, Point<T,N> value) { return p.gt(value); });
    cls.def("ge", [](PointBase<T,N> &p, Point<T,N> value) { return p.ge(value); });
    cls.def("eq", [](PointBase<T,N> &p, T value) { return p.eq(value); });
    cls.def("ne", [](PointBase<T,N> &p, T value) { return p.ne(value); });
    cls.def("lt", [](PointBase<T,N> &p, T value) { return p.lt(value); });
    cls.def("le", [](PointBase<T,N> &p, T value) { return p.le(value); });
    cls.def("gt", [](PointBase<T,N> &p, T value) { return p.gt(value); });
    cls.def("ge", [](PointBase<T,N> &p, T value) { return p.ge(value); });
    cls.def("__str__", &PointBase<T,N>::toString);

    /* Members */
    cls.def("asExtent", &PointBase<T,N>::asExtent);
    cls.def("shift", &PointBase<T,N>::shift);
    cls.def("scale", &PointBase<T,N>::scale);
    cls.def("distanceSquared", &PointBase<T,N>::distanceSquared);
    cls.def("toString", &PointBase<T,N>::toString);

    return cls;
}

// Common functionality
template <typename T, int N>
py::class_<Point<T,N>> declarePoint(py::module &mod, const std::string & suffix) {
    const std::string name = "Point" + suffix;
    py::class_<Point<T,N>> cls(mod, name.c_str(), py::base<PointBase<T, N>>());

    /* Constructors */
    cls.def(py::init<T>(),
        py::arg("val")=static_cast<T>(0));
    // Note that we can't use T here because both types are needed
    cls.def(py::init<Point<double,N> const &>());
    cls.def(py::init<Point<int,N> const &>());

    /* Operators */
    cls.def(py::self + Extent<double,N>());
    cls.def(py::self + Extent<int,N>());
    cls.def("__sub__", [](Point<T,N> &p, Point<T,N> &o) { return p - o; });
    cls.def("__sub__", [](Point<T,N> &p, Extent<T,N> &o) { return p - o; });
    cls.def("__sub__", [](Point<T,N> &p, Point<double,N> &o) { return p - o; });
    cls.def("__sub__", [](Point<T,N> &p, Point<int,N> &o) { return p - o; });
    cls.def("__sub__", [](Point<T,N> &p, Extent<double,N> &o) { return p - o; });
    cls.def("__sub__", [](Point<T,N> &p, Extent<int,N> &o) { return p - o; });
    cls.def("__eq__", [](Point<T,N> &p, Point<T,N> &o) { return p == o; });
    cls.def("__eq__", [](Point<T,N> &p, py::none) { return false; });
    cls.def("__neq__", [](Point<T,N> &p, Point<T,N> &o) { return p != o; });
    cls.def("__neq__", [](Point<T,N> &p, py::none) { return true; });

    /* Members */
    cls.def("clone", [](Point<T,N> &p) {return Point<T,N>{p};});

    return cls;
}

// Add functionality only found in N=2 Points
template <typename T>
py::class_<Point<T,2>> declarePoint2(py::module &mod, const std::string & suffix) {
    auto cls = declarePoint<T,2>(mod, std::string("2") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* self */) { return 2; });

    /* Constructors */
    cls.def(py::init<int, int>());
    cls.def(py::init<double, double>());

    /* Members */
    cls.def("getX", [](Point<T,2> &e) { return e[0]; });
    cls.def("getY", [](Point<T,2> &e) { return e[1]; });
    cls.def("setX", [](Point<T,2> &e, T value) { e[0] = value; });
    cls.def("setY", [](Point<T,2> &e, T value) { e[1] = value; });

    return cls;
}

// Add functionality only found in N=3 Points
template <typename T>
py::class_<Point<T,3>> declarePoint3(py::module &mod, const std::string & suffix) {
    auto cls = declarePoint<T,3>(mod, std::string("3") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* self */) { return 3; });

    /* Constructors */
    cls.def(py::init<int, int, int>());
    cls.def(py::init<double, double, double>());

    /* Members */
    cls.def("getX", [](Point<T,3> &e) { return e[0]; });
    cls.def("getY", [](Point<T,3> &e) { return e[1]; });
    cls.def("getZ", [](Point<T,3> &e) { return e[2]; });
    cls.def("setX", [](Point<T,3> &e, T value) { e[0] = value; });
    cls.def("setY", [](Point<T,3> &e, T value) { e[1] = value; });
    cls.def("setZ", [](Point<T,3> &e, T value) { e[2] = value; });

    return cls;
}

PYBIND11_PLUGIN(_point) {
    py::module mod("_point", "Python wrapper for afw _point library");

    // First declare the bases
    auto clsPointBase2I = declarePointBase<int, 2>(mod, "2I");
    auto clsPointBase3I = declarePointBase<int, 3>(mod, "3I");
    auto clsPointBase2D = declarePointBase<double, 2>(mod, "2D");
    auto clsPointBase3D = declarePointBase<double, 3>(mod, "3D");

    // And then the types
    auto clsPoint2I = declarePoint2<int>(mod, "I");
    auto clsPoint3I = declarePoint3<int>(mod, "I");
    auto clsPoint2D = declarePoint2<double>(mod, "D");
    auto clsPoint3D = declarePoint3<double>(mod, "D");

    /* Operators */
    clsPoint2I.def("__iadd__", [](Point<int,2> &p, Extent<int,2> &o) { p += o; return &p; });
    clsPoint3I.def("__iadd__", [](Point<int,3> &p, Extent<int,3> &o) { p += o; return &p; });
    clsPoint2D.def("__iadd__", [](Point<double,2> &p, Extent<int,2> &o) { p += o; return &p; });
    clsPoint3D.def("__iadd__", [](Point<double,3> &p, Extent<int,3> &o) { p += o; return &p; });
    clsPoint2D.def("__iadd__", [](Point<double,2> &p, Extent<double,2> &o) { p += o; return &p; });
    clsPoint3D.def("__iadd__", [](Point<double,3> &p, Extent<double,3> &o) { p += o; return &p; });

    clsPoint2I.def("__isub__", [](Point<int,2> &p, Extent<int,2> &o) { p -= o; return &p; });
    clsPoint3I.def("__isub__", [](Point<int,3> &p, Extent<int,3> &o) { p -= o; return &p; });
    clsPoint2D.def("__isub__", [](Point<double,2> &p, Extent<int,2> &o) { p -= o; return &p; });
    clsPoint3D.def("__isub__", [](Point<double,3> &p, Extent<int,3> &o) { p -= o; return &p; });
    clsPoint2D.def("__isub__", [](Point<double,2> &p, Extent<double,2> &o) { p -= o; return &p; });
    clsPoint3D.def("__isub__", [](Point<double,3> &p, Extent<double,3> &o) { p -= o; return &p; });

    return mod.ptr();
}

