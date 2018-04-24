/*
 * LSST Data Management System
 * Copyright 2008-2017  LSST/AURA.
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
#include "ndarray/pybind11.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/CoordinateBase.h"
#include "lsst/afw/geom/CoordinateExpr.h"
#include "lsst/utils/python.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

template <typename Derived, typename T, int N>
using PyCoordinateBase = py::class_<CoordinateBase<Derived, T, N>>;

template <int N>
using PyCoordinateExpr = py::class_<CoordinateExpr<N>, CoordinateBase<CoordinateExpr<N>, bool, N>>;

template <typename T, int N>
using PyExtentBase = py::class_<ExtentBase<T, N>, CoordinateBase<Extent<T, N>, T, N>>;

template <typename T, int N>
using PyExtent = py::class_<Extent<T, N>, ExtentBase<T, N>>;

template <typename T, int N>
using PyPointBase = py::class_<PointBase<T, N>, CoordinateBase<Point<T, N>, T, N>>;

template <typename T, int N>
using PyPoint = py::class_<Point<T, N>, PointBase<T, N>>;

template <typename Derived, typename T, int N>
void declareCoordinateBase(py::module &mod, std::string const &suffix) {
    static std::string const name = "CoordinateBase" + suffix;
    PyCoordinateBase<Derived, T, N> cls(mod, name.c_str());

    /* Operators */
    cls.def("__getitem__", [](CoordinateBase<Derived, T, N> &self, int i) -> T {
        return self[utils::python::cppIndex(N, i)];
    });
    cls.def("__setitem__", [](CoordinateBase<Derived, T, N> &self, int i, T value) {
        self[utils::python::cppIndex(N, i)] = value;
    });
    cls.def("__len__", [](CoordinateBase<Derived, T, N> &c) -> int { return N; });
}

template <int N>
void declareCoordinateExpr(py::module &mod, std::string const &suffix) {
    static std::string const name = "CoordinateExpr" + suffix;
    declareCoordinateBase<CoordinateExpr<N>, bool, N>(mod, name);

    PyCoordinateExpr<N> clsCoordinateExpr(mod, name.c_str());

    /* Constructors */
    clsCoordinateExpr.def(py::init<bool>(), "val"_a = false);

    /* Operators */
    clsCoordinateExpr.def("and_", &CoordinateExpr<N>::and_);
    clsCoordinateExpr.def("or_", &CoordinateExpr<N>::or_);
    clsCoordinateExpr.def("not_", &CoordinateExpr<N>::not_);

    mod.def("all", all<N>);
    mod.def("any", any<N>);
}

template <typename T, int N>
void declareExtentBase(py::module &mod, std::string const &suffix) {
    declareCoordinateBase<Extent<T, N>, T, N>(mod, "Extent" + suffix);
    static std::string const name = "ExtentBase" + suffix;

    PyExtentBase<T, N> cls(mod, name.c_str());

    // These are not the usual Python double-underscore operators - they do elementwise comparisons,
    // returning a CoordinateExpr object with boolean x and y values.  NumPy examples to the contrary
    // notwithstanding, true Python comparison operators are expected to return scalar bools.
    cls.def("eq", [](ExtentBase<T, N> const &self, Extent<T, N> other) { return self.eq(other); });
    cls.def("ne", [](ExtentBase<T, N> const &self, Extent<T, N> other) { return self.ne(other); });
    cls.def("lt", [](ExtentBase<T, N> const &self, Extent<T, N> other) { return self.lt(other); });
    cls.def("le", [](ExtentBase<T, N> const &self, Extent<T, N> other) { return self.le(other); });
    cls.def("gt", [](ExtentBase<T, N> const &self, Extent<T, N> other) { return self.gt(other); });
    cls.def("ge", [](ExtentBase<T, N> const &self, Extent<T, N> other) { return self.ge(other); });
    cls.def("eq", [](ExtentBase<T, N> const &self, T other) { return self.eq(other); });
    cls.def("ne", [](ExtentBase<T, N> const &self, T other) { return self.ne(other); });
    cls.def("lt", [](ExtentBase<T, N> const &self, T other) { return self.lt(other); });
    cls.def("le", [](ExtentBase<T, N> const &self, T other) { return self.le(other); });
    cls.def("gt", [](ExtentBase<T, N> const &self, T other) { return self.gt(other); });
    cls.def("ge", [](ExtentBase<T, N> const &self, T other) { return self.ge(other); });

    /* Members */
    cls.def("asPoint", &ExtentBase<T, N>::asPoint);
    cls.def("computeNorm", &ExtentBase<T, N>::computeNorm);
    cls.def("computeSquaredNorm", &ExtentBase<T, N>::computeSquaredNorm);
}

// Common functionality for all Extents, including declaring base classes.
template <typename T, int N>
PyExtent<T, N> declareExtent(py::module &mod, std::string const &suffix) {
    declareExtentBase<T, N>(mod, suffix);
    static std::string const name = "Extent" + suffix;

    PyExtent<T, N> cls(mod, name.c_str());

    /* Constructors */
    cls.def(py::init<T>(), "value"_a = static_cast<T>(0));
    cls.def(py::init<Point<int, N> const &>());
    cls.def(py::init<Point<T, N> const &>());
    cls.def(py::init<Extent<int, N> const &>());
    cls.def(py::init<Extent<T, N> const &>());
    cls.def(py::init<typename Extent<T, N>::EigenVector>());

    /* Operators */
    cls.def("__neg__", [](Extent<T, N> const &self) { return -self; });
    cls.def("__pos__", [](Extent<T, N> const &self) { return self; });
    cls.def("__mul__", [](Extent<T, N> const &self, int other) { return self * other; }, py::is_operator());
    cls.def("__mul__", [](Extent<T, N> const &self, double other) { return self * other; },
            py::is_operator());
    cls.def("__rmul__", [](Extent<T, N> const &self, int other) { return self * other; }, py::is_operator());
    cls.def("__rmul__", [](Extent<T, N> const &self, double other) { return self * other; },
            py::is_operator());
    cls.def("__add__", [](Extent<T, N> const &self, Extent<int, N> const &other) { return self + other; },
            py::is_operator());
    cls.def("__add__", [](Extent<T, N> const &self, Extent<double, N> const &other) { return self + other; },
            py::is_operator());
    cls.def("__add__",
            [](Extent<T, N> const &self, Point<int, N> const &other) { return self + Point<T, N>(other); },
            py::is_operator());
    cls.def("__add__", [](Extent<T, N> const &self, Point<double, N> const &other) { return self + other; },
            py::is_operator());
    cls.def("__sub__",
            [](Extent<T, N> const &self, Extent<int, N> const &other) { return self - Extent<T, N>(other); },
            py::is_operator());
    cls.def("__sub__", [](Extent<T, N> const &self, Extent<double, N> const &other) { return self - other; },
            py::is_operator());
    cls.def("__eq__", [](Extent<T, N> const &self, Extent<T, N> const &other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](Extent<T, N> const &self, Extent<T, N> const &other) { return self != other; },
            py::is_operator());

    /* Members */
    cls.def("clone", [](Extent<T, N> const &self) { return Extent<T, N>{self}; });

    return cls;
}

// Add functionality only found in N=2 Extents (and delgate to declareExtent for the rest)
template <typename T>
PyExtent<T, 2> declareExtent2(py::module &mod, std::string const &suffix) {
    auto cls = declareExtent<T, 2>(mod, std::string("2") + suffix);

    /* Members types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* cls */) { return 2; });

    /* Constructors */
    cls.def(py::init<int, int>(), "x"_a, "y"_a);
    cls.def(py::init<double, double>(), "x"_a, "y"_a);

    /* Members */
    cls.def("getX", [](Extent<T, 2> const &self) { return self[0]; });
    cls.def("getY", [](Extent<T, 2> const &self) { return self[1]; });
    cls.def("setX", [](Extent<T, 2> &self, T other) { self[0] = other; });
    cls.def("setY", [](Extent<T, 2> &self, T other) { self[1] = other; });

    return cls;
}

// Add functionality only found in N=3 Extents (and delgate to declareExtent for the rest)
template <typename T>
PyExtent<T, 3> declareExtent3(py::module &mod, const std::string &suffix) {
    auto cls = declareExtent<T, 3>(mod, std::string("3") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* cls */) { return 3; });

    /* Constructors */
    cls.def(py::init<int, int, int>(), "x"_a, "y"_a, "z"_a);
    cls.def(py::init<double, double, double>(), "x"_a, "y"_a, "z"_a);

    /* Members */
    cls.def("getX", [](Extent<T, 3> const &self) { return self[0]; });
    cls.def("getY", [](Extent<T, 3> const &self) { return self[1]; });
    cls.def("getZ", [](Extent<T, 3> const &self) { return self[2]; });
    cls.def("setX", [](Extent<T, 3> &self, T other) { self[0] = other; });
    cls.def("setY", [](Extent<T, 3> &self, T other) { self[1] = other; });
    cls.def("setZ", [](Extent<T, 3> &self, T other) { self[2] = other; });

    return cls;
}

// Declare mixed-type and type-overloaded operators for Extent with dimension
// N for both int and double. Because pybind11 tries operators (like any
// overload) `in order', int has to come before double in any overloaded
// operators that dispatch on a scalar, and hence they have to be defined here
// instead of declareExtent.
template <int N>
void declareExtentOperators(py::module &mod, PyExtent<int, N> &clsI, PyExtent<double, N> &clsD) {
    // Python's integer division works differently than C++'s for negative numbers - Python
    // uses floor (rounds towards more negative), while C++ truncates (rounds towards zero).
    // Therefore one needs to be careful in the definition of division operators.
    clsI.def("__floordiv__",
             [](Extent<int, N> const &self, int other) -> Extent<int, N> {
                 return floor(self / static_cast<double>(other));
             },
             py::is_operator());

    clsI.def("__truediv__", [](Extent<int, N> const &self, double other) { return self / other; },
             py::is_operator());
    clsD.def("__truediv__", [](Extent<double, N> const &self, double other) { return self / other; },
             py::is_operator());

    clsI.def("__ifloordiv__", [](Extent<int, N> &self, int other) -> Extent<int, N> & {
        self = floor(self / static_cast<double>(other));
        return self;
    });

    clsI.def("__itruediv__", [](Extent<int, N> &self, double other) {
        PyErr_SetString(PyExc_TypeError, "In-place true division not supported for Extent<int,N>.");
        throw py::error_already_set();
    });
    clsD.def("__itruediv__", [](Extent<double, N> &self, double other) -> Extent<double, N> & {
        self /= other;
        return self;
    });

    clsI.def("__iadd__", [](Extent<int, N> &self, Extent<int, N> const &other) -> Extent<int, N> & {
        self += other;
        return self;
    });
    clsD.def("__iadd__", [](Extent<double, N> &self, Extent<double, N> const &other) -> Extent<double, N> & {
        self += other;
        return self;
    });
    clsD.def("__iadd__", [](Extent<double, N> &self, Extent<int, N> const &other) -> Extent<double, N> & {
        self += other;
        return self;
    });

    clsI.def("__isub__", [](Extent<int, N> &self, Extent<int, N> const &other) -> Extent<int, N> & {
        self -= other;
        return self;
    });
    clsD.def("__isub__", [](Extent<double, N> &self, Extent<double, N> const &other) -> Extent<double, N> & {
        self -= other;
        return self;
    });
    clsD.def("__isub__", [](Extent<double, N> &self, Extent<int, N> const &other) -> Extent<double, N> & {
        self -= other;
        return self;
    });

    clsI.def("__imul__", [](Extent<int, N> &self, int other) -> Extent<int, N> & {
        self *= other;
        return self;
    });
    clsD.def("__imul__", [](Extent<double, N> &self, int other) -> Extent<double, N> & {
        self *= other;
        return self;
    });
    clsD.def("__imul__", [](Extent<double, N> &self, double other) -> Extent<double, N> & {
        self *= other;
        return self;
    });

    // Operator-like free functions
    mod.def("truncate", truncate<N>);
    mod.def("floor", floor<N>);
    mod.def("ceil", ceil<N>);
    // And method versions, since Python doens't have ADL
    clsD.def("truncate", truncate<N>);
    clsD.def("floor", floor<N>);
    clsD.def("ceil", ceil<N>);
}

template <typename T, int N>
void declarePointBase(py::module &mod, std::string const &suffix) {
    declareCoordinateBase<Point<T, N>, T, N>(mod, "Point" + suffix);
    static std::string const name = "PointBase" + suffix;
    PyPointBase<T, N> cls(mod, name.c_str());

    // These are not the usual Python double-underscore operators - they do elementwise comparisons,
    // returning a CoordinateExpr object with boolean x and y values.  NumPy examples to the contrary
    // notwithstanding, true Python comparison operators are expected to return scalar bools.
    cls.def("eq", [](PointBase<T, N> const &self, Point<T, N> const &other) { return self.eq(other); });
    cls.def("ne", [](PointBase<T, N> const &self, Point<T, N> const &other) { return self.ne(other); });
    cls.def("lt", [](PointBase<T, N> const &self, Point<T, N> const &other) { return self.lt(other); });
    cls.def("le", [](PointBase<T, N> const &self, Point<T, N> const &other) { return self.le(other); });
    cls.def("gt", [](PointBase<T, N> const &self, Point<T, N> const &other) { return self.gt(other); });
    cls.def("ge", [](PointBase<T, N> const &self, Point<T, N> const &other) { return self.ge(other); });
    cls.def("eq", [](PointBase<T, N> const &self, T other) { return self.eq(other); });
    cls.def("ne", [](PointBase<T, N> const &self, T other) { return self.ne(other); });
    cls.def("lt", [](PointBase<T, N> const &self, T other) { return self.lt(other); });
    cls.def("le", [](PointBase<T, N> const &self, T other) { return self.le(other); });
    cls.def("gt", [](PointBase<T, N> const &self, T other) { return self.gt(other); });
    cls.def("ge", [](PointBase<T, N> const &self, T other) { return self.ge(other); });

    /* Members */
    cls.def("asExtent", &PointBase<T, N>::asExtent);
    cls.def("shift", &PointBase<T, N>::shift);
    cls.def("scale", &PointBase<T, N>::scale);
    cls.def("distanceSquared", &PointBase<T, N>::distanceSquared);
    cls.def("toString", &PointBase<T, N>::toString);
}

// Common functionality
template <typename T, int N>
PyPoint<T, N> declarePoint(py::module &mod, std::string const &suffix) {
    declarePointBase<T, N>(mod, suffix);
    static std::string const name = "Point" + suffix;
    PyPoint<T, N> cls(mod, name.c_str());

    /* Constructors */
    cls.def(py::init<T>(), "value"_a = static_cast<T>(0));
    // Note that we can't use T here because both types are needed
    cls.def(py::init<Point<double, N> const &>());
    cls.def(py::init<Point<int, N> const &>());
    cls.def(py::init<Extent<T, N> const &>());
    cls.def(py::init<typename Point<T, N>::EigenVector>());

    /* Operators */
    cls.def("__add__", [](Point<T, N> const &self, Extent<double, N> &other) { return self + other; },
            py::is_operator());
    cls.def("__add__", [](Point<T, N> const &self, Extent<int, N> &other) { return self + other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> const &self, Point<T, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> const &self, Extent<T, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> const &self, Point<double, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> const &self, Point<int, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> const &self, Extent<double, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__sub__", [](Point<T, N> const &self, Extent<int, N> &other) { return self - other; },
            py::is_operator());
    cls.def("__eq__", [](Point<T, N> const &self, Point<T, N> const &other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](Point<T, N> const &self, Point<T, N> const &other) { return self != other; },
            py::is_operator());

    /* Members */
    cls.def("clone", [](Point<T, N> const &self) { return Point<T, N>{self}; });

    return cls;
}

// Add functionality only found in N=2 Points
template <typename T>
PyPoint<T, 2> declarePoint2(py::module &mod, std::string const &suffix) {
    auto cls = declarePoint<T, 2>(mod, std::string("2") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* cls */) { return 2; });

    /* Constructors */
    cls.def(py::init<int, int>(), "x"_a, "y"_a);
    cls.def(py::init<double, double>(), "x"_a, "y"_a);

    /* Members */
    cls.def("getX", [](Point<T, 2> const &self) { return self[0]; });
    cls.def("getY", [](Point<T, 2> const &self) { return self[1]; });
    cls.def("setX", [](Point<T, 2> &self, T other) { self[0] = other; });
    cls.def("setY", [](Point<T, 2> &self, T other) { self[1] = other; });
    return cls;
}

// Add functionality only found in N=3 Points
template <typename T>
PyPoint<T, 3> declarePoint3(py::module &mod, std::string const &suffix) {
    auto cls = declarePoint<T, 3>(mod, std::string("3") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* cls */) { return 3; });

    /* Constructors */
    cls.def(py::init<int, int, int>(), "x"_a, "y"_a, "z"_a);
    cls.def(py::init<double, double, double>(), "x"_a, "y"_a, "z"_a);

    /* Members */
    cls.def("getX", [](Point<T, 3> const &self) { return self[0]; });
    cls.def("getY", [](Point<T, 3> const &self) { return self[1]; });
    cls.def("getZ", [](Point<T, 3> const &self) { return self[2]; });
    cls.def("setX", [](Point<T, 3> &self, T other) { self[0] = other; });
    cls.def("setY", [](Point<T, 3> &self, T other) { self[1] = other; });
    cls.def("setZ", [](Point<T, 3> &self, T other) { self[2] = other; });

    return cls;
}

// Declare mixed-type and type-overloaded operators for Point with dimension
// N for both int and double. Because pybind11 tries operators (like any
// overload) `in order', int has to come before double in any overloaded
// operators that dispatch on a scalar, and hence they have to be defined here
// instead of declareExtent.
template <int N>
void declarePointOperators(py::module &mod, PyPoint<int, N> &clsI, PyPoint<double, N> &clsD) {
    clsI.def("__iadd__", [](Point<int, N> &self, Extent<int, N> const &other) {
        self += other;
        return &self;
    });
    clsD.def("__iadd__", [](Point<double, N> &self, Extent<int, N> const &other) {
        self += other;
        return &self;
    });
    clsD.def("__iadd__", [](Point<double, N> &self, Extent<double, N> const &other) {
        self += other;
        return &self;
    });
    clsI.def("__isub__", [](Point<int, N> &self, Extent<int, N> const &other) {
        self -= other;
        return &self;
    });
    clsD.def("__isub__", [](Point<double, N> &self, Extent<int, N> const &other) {
        self -= other;
        return &self;
    });
    clsD.def("__isub__", [](Point<double, N> &self, Extent<double, N> const &other) {
        self -= other;
        return &self;
    });
}

PYBIND11_PLUGIN(coordinates) {
    py::module mod("coordinates");

    // Only the interface-level classes are defined here, and these functions
    // call others to define their base classes, since those are never shared.

    declareCoordinateExpr<2>(mod, "2");
    declareCoordinateExpr<3>(mod, "3");

    auto clsExtent2I = declareExtent2<int>(mod, "I");
    auto clsExtent2D = declareExtent2<double>(mod, "D");
    declareExtentOperators(mod, clsExtent2I, clsExtent2D);

    auto clsExtent3I = declareExtent3<int>(mod, "I");
    auto clsExtent3D = declareExtent3<double>(mod, "D");
    declareExtentOperators(mod, clsExtent3I, clsExtent3D);

    auto clsPoint2I = declarePoint2<int>(mod, "I");
    auto clsPoint2D = declarePoint2<double>(mod, "D");
    declarePointOperators(mod, clsPoint2I, clsPoint2D);

    auto clsPoint3I = declarePoint3<int>(mod, "I");
    auto clsPoint3D = declarePoint3<double>(mod, "D");
    declarePointOperators(mod, clsPoint3I, clsPoint3D);

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::geom::<anonymous>
