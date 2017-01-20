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
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"

namespace py = pybind11;

using namespace lsst::afw::geom;

template <typename T, int N>
py::class_<ExtentBase<T,N>> declareExtentBase(py::module &mod, const std::string & suffix) {
    const std::string name = "ExtentBase" + suffix;
    py::class_<ExtentBase<T,N>, CoordinateBase<Extent<T,N>,T,N>> cls(mod, name.c_str());

    /* Operators */
    cls.def("eq", [](ExtentBase<T,N> &p, Extent<T,N> value) { return p.eq(value); });
    cls.def("ne", [](ExtentBase<T,N> &p, Extent<T,N> value) { return p.ne(value); });
    cls.def("lt", [](ExtentBase<T,N> &p, Extent<T,N> value) { return p.lt(value); });
    cls.def("le", [](ExtentBase<T,N> &p, Extent<T,N> value) { return p.le(value); });
    cls.def("gt", [](ExtentBase<T,N> &p, Extent<T,N> value) { return p.gt(value); });
    cls.def("ge", [](ExtentBase<T,N> &p, Extent<T,N> value) { return p.ge(value); });
    cls.def("eq", [](ExtentBase<T,N> &p, T value) { return p.eq(value); });
    cls.def("ne", [](ExtentBase<T,N> &p, T value) { return p.ne(value); });
    cls.def("lt", [](ExtentBase<T,N> &p, T value) { return p.lt(value); });
    cls.def("le", [](ExtentBase<T,N> &p, T value) { return p.le(value); });
    cls.def("gt", [](ExtentBase<T,N> &p, T value) { return p.gt(value); });
    cls.def("ge", [](ExtentBase<T,N> &p, T value) { return p.ge(value); });

    /* Members */
    cls.def("asPoint", &ExtentBase<T,N>::asPoint);
    cls.def("computeNorm", &ExtentBase<T,N>::computeNorm);
    cls.def("computeSquaredNorm", &ExtentBase<T,N>::computeSquaredNorm);

    return cls;
}

// Common functionality
template <typename T, int N>
py::class_<Extent<T,N>> declareExtent(py::module &mod, const std::string & suffix) {
    const std::string name = "Extent" + suffix;
    py::class_<Extent<T,N>, ExtentBase<T, N>> cls(mod, name.c_str());

    /* Constructors */
    cls.def(py::init<T>(),
        py::arg("val")=static_cast<T>(0));
    cls.def(py::init<Point<int,N> const &>());
    cls.def(py::init<Point<T,N> const &>());
    cls.def(py::init<Extent<int,N> const &>());
    cls.def(py::init<Extent<T,N> const &>());
    cls.def(py::init<typename Extent<T, N>::EigenVector>());

    /* Operators */
    cls.def(-py::self);
    cls.def(+py::self);
    cls.def("__mul__", [](Extent<T,N> &e, int val) { return e * val; }, py::is_operator());
    cls.def("__mul__", [](Extent<T,N> &e, double val) { return e * val; }, py::is_operator());
    cls.def("__rmul__", [](Extent<T,N> &e, int val) { return e * val; }, py::is_operator());
    cls.def("__rmul__", [](Extent<T,N> &e, double val) { return e * val; }, py::is_operator());
    cls.def("__add__", [](Extent<T,N> &e, Extent<int,N> &o) { return e + o; }, py::is_operator());
    cls.def("__add__", [](Extent<T,N> &e, Extent<double,N> &o) { return e + o; }, py::is_operator());
    cls.def("__add__", [](Extent<T,N> &e, Point<int,N> &o) { return e + Point<T,N>(o); }, py::is_operator());
    cls.def("__add__", [](Extent<T,N> &e, Point<double,N> &o) { return e + o; }, py::is_operator());
    cls.def("__sub__", [](Extent<T,N> &e, Extent<int,N> &o) { return e - Extent<T,N>(o); }, py::is_operator());
    cls.def("__sub__", [](Extent<T,N> &e, Extent<double,N> &o) { return e - o; }, py::is_operator());
    cls.def("__eq__", [](Extent<T,N> &e, Extent<T,N> &o) { return e == o; }, py::is_operator());
    cls.def("__eq__", [](Extent<T,N> &e, py::none) { return false; }, py::is_operator());
    cls.def("__ne__", [](Extent<T,N> &e, Extent<T,N> &o) { return e != o; }, py::is_operator());
    cls.def("__ne__", [](Extent<T,N> &e, py::none) { return true; }, py::is_operator());

    /* Members */
    cls.def("clone", [](Extent<T,N> &p) {return Extent<T,N>{p};});

    return cls;
}

// Add functionality only found in N=2 Extents
template <typename T>
py::class_<Extent<T,2>> declareExtent2(py::module &mod, const std::string & suffix) {
    auto cls = declareExtent<T,2>(mod, std::string("2") + suffix);

    /* Members types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* self */) { return 2; });

    /* Constructors */
    cls.def(py::init<int, int>());
    cls.def(py::init<double, double>());

    /* Members */
    cls.def("getX", [](Extent<T,2> &e) { return e[0]; });
    cls.def("getY", [](Extent<T,2> &e) { return e[1]; });
    cls.def("setX", [](Extent<T,2> &e, T value) { e[0] = value; });
    cls.def("setY", [](Extent<T,2> &e, T value) { e[1] = value; });

    return cls;
}

// Add functionality only found in N=3 Extents
template <typename T>
py::class_<Extent<T,3>> declareExtent3(py::module &mod, const std::string & suffix) {
    auto cls = declareExtent<T,3>(mod, std::string("3") + suffix);

    /* Member types and enums */
    cls.def_property_readonly_static("dimensions", [](py::object /* self */) { return 3; });

    /* Constructors */
    cls.def(py::init<int, int, int>());
    cls.def(py::init<double, double, double>());

    /* Members */
    cls.def("getX", [](Extent<T,3> &e) { return e[0]; });
    cls.def("getY", [](Extent<T,3> &e) { return e[1]; });
    cls.def("getZ", [](Extent<T,3> &e) { return e[2]; });
    cls.def("setX", [](Extent<T,3> &e, T value) { e[0] = value; });
    cls.def("setY", [](Extent<T,3> &e, T value) { e[1] = value; });
    cls.def("setZ", [](Extent<T,3> &e, T value) { e[2] = value; });

    return cls;
}

PYBIND11_PLUGIN(_extent) {
    py::module mod("_extent", "Python wrapper for afw _extent library");

    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    };

    // First declare the bases
    auto clsExtentBase2I = declareExtentBase<int, 2>(mod, "2I");
    auto clsExtentBase3I = declareExtentBase<int, 3>(mod, "3I");
    auto clsExtentBase2D = declareExtentBase<double, 2>(mod, "2D");
    auto clsExtentBase3D = declareExtentBase<double, 3>(mod, "3D");

    // And then the types
    auto clsExtent2I = declareExtent2<int>(mod, "I");
    auto clsExtent3I = declareExtent3<int>(mod, "I");
    auto clsExtent2D = declareExtent2<double>(mod, "D");
    auto clsExtent3D = declareExtent3<double>(mod, "D");

    /* Operators */

    // Python's integer division works differently than C++'s for negative numbers - Python
    // uses floor (rounds towards more negative), while C++ truncates (rounds towards zero).
    // Therefore one needs to be carefull in the definition of division operators.
    // Also note that pybind11 tries operators (like any overload) `in order'. So int has
    // to come before double if specialization is needed.
    clsExtent2I.def("__floordiv__", [](Extent<int,2> &e, int val) -> Extent<int,2> { return floor(e / static_cast<double>(val)); });
    clsExtent3I.def("__floordiv__", [](Extent<int,3> &e, int val) -> Extent<int,3> { return floor(e / static_cast<double>(val)); });

    clsExtent2I.def("__div__", [](Extent<int,2> &e, int val) { return floor(e / static_cast<double>(val)); });
    clsExtent3I.def("__div__", [](Extent<int,3> &e, int val) { return floor(e / static_cast<double>(val)); });
    clsExtent2I.def("__div__", [](Extent<int,2> &e, double val) { return e / val; });
    clsExtent3I.def("__div__", [](Extent<int,3> &e, double val) { return e / val; });
    clsExtent2D.def("__div__", [](Extent<double,2> &e, double val) { return e / val; });
    clsExtent3D.def("__div__", [](Extent<double,3> &e, double val) { return e / val; });

    clsExtent2I.def("__truediv__", [](Extent<int,2> &e, double val) { return e / val; });
    clsExtent3I.def("__truediv__", [](Extent<int,3> &e, double val) { return e / val; });
    clsExtent2D.def("__truediv__", [](Extent<double,2> &e, double val) { return e / val; });
    clsExtent3D.def("__truediv__", [](Extent<double,3> &e, double val) { return e / val; });

    clsExtent2I.def("__ifloordiv__", [](Extent<int,2> &e, int val) -> Extent<int,2>& {  e = floor(e / static_cast<double>(val)); return e;});
    clsExtent3I.def("__ifloordiv__", [](Extent<int,3> &e, int val) -> Extent<int,3>& {  e = floor(e / static_cast<double>(val)); return e;});

    clsExtent2I.def("__idiv__", [](Extent<int,2> &e, int val) -> Extent<int,2>& {  e = floor(e / static_cast<double>(val)); return e;});
    clsExtent3I.def("__idiv__", [](Extent<int,3> &e, int val) -> Extent<int,3>& {  e = floor(e / static_cast<double>(val)); return e;});
    clsExtent2D.def("__idiv__", [](Extent<double,2> &e, double val) -> Extent<double,2>& {  e /= val; return e;});
    clsExtent3D.def("__idiv__", [](Extent<double,3> &e, double val) -> Extent<double,3>& {  e /= val; return e;});

    clsExtent2I.def("__itruediv__", [](Extent<int,2> &e, double val) {
        PyErr_SetString(PyExc_TypeError, "In-place true division not supported for Extent<int,N>.");
        throw py::error_already_set();
    });
    clsExtent3I.def("__itruediv__", [](Extent<int,3> &e, double val) {
        PyErr_SetString(PyExc_TypeError, "In-place true division not supported for Extent<int,N>.");
        throw py::error_already_set();
    });
    clsExtent2D.def("__itruediv__", [](Extent<double,2> &e, double val) -> Extent<double,2>& {  e /= val; return e;});
    clsExtent3D.def("__itruediv__", [](Extent<double,3> &e, double val) -> Extent<double,3>& {  e /= val; return e;});

    clsExtent2I.def("__iadd__", [](Extent<int,2> &lhs, Extent<int,2> &rhs) -> Extent<int,2>& { lhs += rhs; return lhs; });
    clsExtent3I.def("__iadd__", [](Extent<int,3> &lhs, Extent<int,3> &rhs) -> Extent<int,3>& { lhs += rhs; return lhs; });
    clsExtent2D.def("__iadd__", [](Extent<double,2> &lhs, Extent<double,2> &rhs) -> Extent<double,2>& { lhs += rhs; return lhs; });
    clsExtent3D.def("__iadd__", [](Extent<double,3> &lhs, Extent<double,3> &rhs) -> Extent<double,3>& { lhs += rhs; return lhs; });
    clsExtent2D.def("__iadd__", [](Extent<double,2> &lhs, Extent<int,2> &rhs) -> Extent<double,2>& { lhs += rhs; return lhs; });
    clsExtent3D.def("__iadd__", [](Extent<double,3> &lhs, Extent<int,3> &rhs) -> Extent<double,3>& { lhs += rhs; return lhs; });

    clsExtent2I.def("__isub__", [](Extent<int,2> &lhs, Extent<int,2> &rhs) -> Extent<int,2>& { lhs -= rhs; return lhs; });
    clsExtent3I.def("__isub__", [](Extent<int,3> &lhs, Extent<int,3> &rhs) -> Extent<int,3>& { lhs -= rhs; return lhs; });
    clsExtent2D.def("__isub__", [](Extent<double,2> &lhs, Extent<double,2> &rhs) -> Extent<double,2>& { lhs -= rhs; return lhs; });
    clsExtent3D.def("__isub__", [](Extent<double,3> &lhs, Extent<double,3> &rhs) -> Extent<double,3>& { lhs -= rhs; return lhs; });
    clsExtent2D.def("__isub__", [](Extent<double,2> &lhs, Extent<int,2> &rhs) -> Extent<double,2>& { lhs -= rhs; return lhs; });
    clsExtent3D.def("__isub__", [](Extent<double,3> &lhs, Extent<int,3> &rhs) -> Extent<double,3>& { lhs -= rhs; return lhs; });

    clsExtent2I.def("__imul__", [](Extent<int,2> &lhs, int rhs) -> Extent<int,2>& { lhs *= rhs; return lhs; });
    clsExtent3I.def("__imul__", [](Extent<int,3> &lhs, int rhs) -> Extent<int,3>& { lhs *= rhs; return lhs; });
    clsExtent2D.def("__imul__", [](Extent<double,2> &lhs, int rhs) -> Extent<double,2>& { lhs *= rhs; return lhs; });
    clsExtent2D.def("__imul__", [](Extent<double,2> &lhs, double rhs) -> Extent<double,2>&  { lhs *= rhs; return lhs; });
    clsExtent3D.def("__imul__", [](Extent<double,3> &lhs, int rhs) -> Extent<double,3>&  { lhs *= rhs; return lhs; });
    clsExtent3D.def("__imul__", [](Extent<double,3> &lhs, double rhs) -> Extent<double,3>&  { lhs *= rhs; return lhs; });

    /* Members */
    mod.def("truncate", truncate<2>);
    mod.def("truncate", truncate<3>);
    mod.def("floor", floor<2>);
    mod.def("floor", floor<3>);
    mod.def("ceil", ceil<2>);
    mod.def("ceil", ceil<3>);

    return mod.ptr();
}
