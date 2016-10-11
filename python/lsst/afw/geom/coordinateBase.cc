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

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/CoordinateBase.h"

namespace py = pybind11;

using namespace lsst::afw::geom;

template <typename Derived, typename T, int N>
py::class_<CoordinateBase<Derived,T,N>> declareCoordinateBase(py::module & mod, const std::string & suffix) {
    const std::string & name = "CoordinateBase" + suffix;
    py::class_<CoordinateBase<Derived,T,N>> cls(mod, name.c_str());

    /* Operators */
    cls.def("__getitem__", [](CoordinateBase<Derived,T,N> &c, int i) -> T {
        if (i < 0 || i >= N) {
            throw py::index_error{};
        } else {
            return c[i];
        }
    });
    cls.def("__setitem__", [](CoordinateBase<Derived,T,N> &c, int i, T value) {
        if (i < 0 || i >= N) {
            throw py::index_error{};
        } else {
            c[i] = value;
        }
    });
    cls.def("__len__", [](CoordinateBase<Derived,T,N> &c) -> int {
        return N;
    });

    return cls;
}

PYBIND11_PLUGIN(_coordinateBase) {
    py::module mod("_coordinateBase", "Python wrapper for afw _coordinateBase library");

    // These declarations are needed as bases of Point, Extent and CoordinateExpr respectively
    auto clsCoordinateBasePoint2I = declareCoordinateBase<Point<int, 2>, int, 2>(mod, "Point2I");
    auto clsCoordinateBasePoint3I = declareCoordinateBase<Point<int, 3>, int, 3>(mod, "Point3I");
    auto clsCoordinateBasePoint2D = declareCoordinateBase<Point<double, 2>, double, 2>(mod, "Point2D");
    auto clsCoordinateBasePoint3D = declareCoordinateBase<Point<double, 3>, double, 3>(mod, "Point3D");
    auto clsCoordinateBaseExtent2I = declareCoordinateBase<Extent<int, 2>, int, 2>(mod, "Extent2I");
    auto clsCoordinateBaseExtent3I = declareCoordinateBase<Extent<int, 3>, int, 3>(mod, "Extent3I");
    auto clsCoordinateBaseExtent2D = declareCoordinateBase<Extent<double, 2>, double, 2>(mod, "Extent2D");
    auto clsCoordinateBaseExtent3D = declareCoordinateBase<Extent<double, 3>, double, 3>(mod, "Extent3D");
    auto clsCoordinateBase2 = declareCoordinateBase<CoordinateExpr<2>, bool, 2>(mod, "2");
    auto clsCoordinateBase3 = declareCoordinateBase<CoordinateExpr<3>, bool, 3>(mod, "3");

    return mod.ptr();
}

