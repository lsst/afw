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
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace geom {

namespace {

template <typename T, int N>
py::class_<PointBase<T, N>> declarePointBase(py::module &mod, const std::string &suffix) {
    const std::string name = "PointBase" + suffix;
    py::class_<PointBase<T, N>, CoordinateBase<Point<T, N>, T, N>> cls(mod, name.c_str());

    /* Operators */
    cls.def("eq", [](PointBase<T, N> &self, Point<T, N> other) { return self.eq(other); });
    cls.def("ne", [](PointBase<T, N> &self, Point<T, N> other) { return self.ne(other); });
    cls.def("lt", [](PointBase<T, N> &self, Point<T, N> other) { return self.lt(other); });
    cls.def("le", [](PointBase<T, N> &self, Point<T, N> other) { return self.le(other); });
    cls.def("gt", [](PointBase<T, N> &self, Point<T, N> other) { return self.gt(other); });
    cls.def("ge", [](PointBase<T, N> &self, Point<T, N> other) { return self.ge(other); });
    cls.def("eq", [](PointBase<T, N> &self, T other) { return self.eq(other); });
    cls.def("ne", [](PointBase<T, N> &self, T other) { return self.ne(other); });
    cls.def("lt", [](PointBase<T, N> &self, T other) { return self.lt(other); });
    cls.def("le", [](PointBase<T, N> &self, T other) { return self.le(other); });
    cls.def("gt", [](PointBase<T, N> &self, T other) { return self.gt(other); });
    cls.def("ge", [](PointBase<T, N> &self, T other) { return self.ge(other); });

    /* Members */
    cls.def("asExtent", &PointBase<T, N>::asExtent);
    cls.def("shift", &PointBase<T, N>::shift);
    cls.def("scale", &PointBase<T, N>::scale);
    cls.def("distanceSquared", &PointBase<T, N>::distanceSquared);
    cls.def("toString", &PointBase<T, N>::toString);

    return cls;
}

// Common functionality
template <typename T, int N>
py::class_<Point<T, N>> declarePoint(py::module &mod, const std::string &suffix) {
    const std::string name = "Point" + suffix;
    py::class_<Point<T, N>, PointBase<T, N>> cls(mod, name.c_str());

    /* Constructors */
    cls.def(py::init<T>(), "value"_a=static_cast<T>(0));
    // Note that we can't use T here because both types are needed
    cls.def(py::init<Point<double, N> const &>());
    cls.def(py::init<Point<int, N> const &>());
    cls.def(py::init<Extent<T, N> const &>());
    cls.def(py::init<typename Point<T, N>::EigenVector>());

    /* Operators */
    cls.def("__add__", [](Point<T, N> &self, Extent<double, N> &other) { return self + other; },
            py::is_operator());
    cls.def("__add__", [](Point<T, N> &self, Extent<int, N> &other) { return self + other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> &self, Point<T, N> &other) { return self - other; }, py::is_operator());
    cls.def("__sub__", [](Point<T, N> &self, Extent<T, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> &self, Point<double, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> &self, Point<int, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> &self, Extent<double, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> &self, Extent<int, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__eq__", [](Point<T, N> &self, Point<T, N> &other) { return self == other; }, py::is_operator());
    cls.def("__eq__", [](Point<T, N> &self, py::none) { return false; }, py::is_operator());
    cls.def("__ne__", [](Point<T, N> &self, Point<T, N> &other) { return self != other; }, py::is_operator());
    cls.def("__ne__", [](Point<T, N> &self, py::none) { return true; }, py::is_operator());

    /* Members */
    cls.def("clone", [](Point<T, N> &self) { return Point<T, N>{self}; });

    return cls;
}

// Add functionality only found in N=2 Points
template <typename T>
py::class_<Point<T, 2>> declarePoint2(py::module &mod, const std::string &suffix) {
    auto cls = declarePoint<T, 2>(mod, std::string("2") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* self */) { return 2; });

    /* Constructors */
    cls.def(py::init<int, int>());
    cls.def(py::init<double, double>());

    /* Members */
    cls.def("getX", [](Point<T, 2> &self) { return self[0]; });
    cls.def("getY", [](Point<T, 2> &self) { return self[1]; });
    cls.def("setX", [](Point<T, 2> &self, T other) { self[0] = other; });
    cls.def("setY", [](Point<T, 2> &self, T other) { self[1] = other; });

    return cls;
}

// Add functionality only found in N=3 Points
template <typename T>
py::class_<Point<T, 3>> declarePoint3(py::module &mod, const std::string &suffix) {
    auto cls = declarePoint<T, 3>(mod, std::string("3") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* self */) { return 3; });

    /* Constructors */
    cls.def(py::init<int, int, int>());
    cls.def(py::init<double, double, double>());

    /* Members */
    cls.def("getX", [](Point<T, 3> &self) { return self[0]; });
    cls.def("getY", [](Point<T, 3> &self) { return self[1]; });
    cls.def("getZ", [](Point<T, 3> &self) { return self[2]; });
    cls.def("setX", [](Point<T, 3> &self, T other) { self[0] = other; });
    cls.def("setY", [](Point<T, 3> &self, T other) { self[1] = other; });
    cls.def("setZ", [](Point<T, 3> &self, T other) { self[2] = other; });

    return cls;
}

}  // namespace lsst::afw::geom::<anonymous>

PYBIND11_PLUGIN(_point) {
    py::module mod("_point", "Python wrapper for afw _point library");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    };

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
    clsPoint2I.def("__iadd__", [](Point<int, 2> &self, Extent<int, 2> &other) {
        self += other;
        return &self;
    });
    clsPoint3I.def("__iadd__", [](Point<int, 3> &self, Extent<int, 3> &other) {
        self += other;
        return &self;
    });
    clsPoint2D.def("__iadd__", [](Point<double, 2> &self, Extent<int, 2> &other) {
        self += other;
        return &self;
    });
    clsPoint3D.def("__iadd__", [](Point<double, 3> &self, Extent<int, 3> &other) {
        self += other;
        return &self;
    });
    clsPoint2D.def("__iadd__", [](Point<double, 2> &self, Extent<double, 2> &other) {
        self += other;
        return &self;
    });
    clsPoint3D.def("__iadd__", [](Point<double, 3> &self, Extent<double, 3> &other) {
        self += other;
        return &self;
    });

    clsPoint2I.def("__isub__", [](Point<int, 2> &self, Extent<int, 2> &other) {
        self -= other;
        return &self;
    });
    clsPoint3I.def("__isub__", [](Point<int, 3> &self, Extent<int, 3> &other) {
        self -= other;
        return &self;
    });
    clsPoint2D.def("__isub__", [](Point<double, 2> &self, Extent<int, 2> &other) {
        self -= other;
        return &self;
    });
    clsPoint3D.def("__isub__", [](Point<double, 3> &self, Extent<int, 3> &other) {
        self -= other;
        return &self;
    });
    clsPoint2D.def("__isub__", [](Point<double, 2> &self, Extent<double, 2> &other) {
        self -= other;
        return &self;
    });
    clsPoint3D.def("__isub__", [](Point<double, 3> &self, Extent<double, 3> &other) {
        self -= other;
        return &self;
    });

    return mod.ptr();
}
}
}
}  // namespace lsst::afw::geom
