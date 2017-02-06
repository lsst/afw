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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "lsst/afw/geom/Box.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

using PyBox2I = py::class_<Box2I>;
using PyBox2D = py::class_<Box2D>;

PYBIND11_PLUGIN(_Box) {
    py::module mod("_Box");

    py::object modCoordinates = py::module::import("lsst.afw.geom._coordinates");

    /* Box2UI */

    PyBox2I clsBox2I(mod, "Box2I");

    clsBox2I.attr("Point") = modCoordinates.attr("Point2I");
    clsBox2I.attr("Extent") = modCoordinates.attr("Extent2I");

    py::enum_<Box2I::EdgeHandlingEnum>(clsBox2I, "EdgeHandlingEnum")
        .value("EXPAND", Box2I::EdgeHandlingEnum::EXPAND)
        .value("SHRINK", Box2I::EdgeHandlingEnum::SHRINK)
        .export_values();

    clsBox2I.def(py::init<>());
    clsBox2I.def(py::init<Point2I const &, Point2I const &, bool>(),
        "minimum"_a, "maximum"_a, "invert"_a=true);
    clsBox2I.def(py::init<Point2I const &, Extent2I const &, bool>(),
        "minimum"_a, "dimensions"_a, "invert"_a=true);
    clsBox2I.def(py::init<Box2D const &, Box2I::EdgeHandlingEnum>(),
        "other"_a, "edgeHandling"_a=Box2I::EXPAND);
    clsBox2I.def(py::init<Box2I const &>(), "other"_a);

    clsBox2I.def("__eq__",
                 [](Box2I const & self, Box2I const & other) { return self == other; },
                 py::is_operator());
    clsBox2I.def("__ne__",
                 [](Box2I const & self, Box2I const & other) { return self != other; },
                 py::is_operator());

    clsBox2I.def("swap", &Box2I::swap);
    clsBox2I.def("getMin", &Box2I::getMin);
    clsBox2I.def("getMinX", &Box2I::getMinX);
    clsBox2I.def("getMinY", &Box2I::getMinY);
    clsBox2I.def("getMax", &Box2I::getMax);
    clsBox2I.def("getMaxX", &Box2I::getMaxX);
    clsBox2I.def("getMaxY", &Box2I::getMaxY);
    clsBox2I.def("getBegin", &Box2I::getBegin);
    clsBox2I.def("getBeginX", &Box2I::getBeginX);
    clsBox2I.def("getBeginY", &Box2I::getBeginY);
    clsBox2I.def("getEnd", &Box2I::getEnd);
    clsBox2I.def("getEndX", &Box2I::getEndX);
    clsBox2I.def("getEndY", &Box2I::getEndY);
    clsBox2I.def("getDimensions", &Box2I::getDimensions);
    clsBox2I.def("getWidth", &Box2I::getWidth);
    clsBox2I.def("getHeight", &Box2I::getHeight);
    clsBox2I.def("getArea", &Box2I::getArea);
    clsBox2I.def("isEmpty", &Box2I::isEmpty);
    clsBox2I.def("contains", (bool (Box2I::*)(Point2I const &) const) &Box2I::contains);
    clsBox2I.def("contains", (bool (Box2I::*)(Box2I const &) const) &Box2I::contains);
    clsBox2I.def("overlaps", &Box2I::overlaps);
    clsBox2I.def("grow", (void (Box2I::*)(int)) &Box2I::grow);
    clsBox2I.def("grow", (void (Box2I::*)(Extent2I const&)) &Box2I::grow);
    clsBox2I.def("shift", &Box2I::shift);
    clsBox2I.def("flipLR", &Box2I::flipLR);
    clsBox2I.def("flipTB", &Box2I::flipTB);
    clsBox2I.def("include", (void (Box2I::*)(Point2I const &)) &Box2I::include);
    clsBox2I.def("include", (void (Box2I::*)(Box2I const &)) &Box2I::include);
    clsBox2I.def("clip", &Box2I::clip);
    clsBox2I.def("getCorners", &Box2I::getCorners);
    clsBox2I.def("toString", &Box2I::toString);
    clsBox2I.def(
        "__repr__",
        [](Box2I const & self) {
            return py::str("Box2D(minimum={}, dimensions={})").format(
                py::repr(py::cast(self.getMin())),
                py::repr(py::cast(self.getDimensions()))
            );
        }
    );
    clsBox2I.def(
        "__str__",
        [](Box2I const & self) {
            return py::str("(minimum={}, maximum={})").format(
                py::str(py::cast(self.getMin())),
                py::str(py::cast(self.getMax()))
            );
        }
    );
    clsBox2I.def(
        "__reduce__",
        [clsBox2I](Box2I const & self) {
            return py::make_tuple(
                clsBox2I,
                make_tuple(py::cast(self.getMin()), py::cast(self.getMax()))
            );
        }
    );
    clsBox2I.def(
        "getSlices",
        [](Box2I const & self) {
            return py::make_tuple(
                py::slice(self.getBeginY(), self.getEndY(), 1),
                py::slice(self.getBeginX(), self.getEndX(), 1)
            );
        }
    );

    /* Box2D */

    PyBox2D clsBox2D(mod, "Box2D");

    clsBox2I.attr("Point") = modCoordinates.attr("Point2D");
    clsBox2I.attr("Extent") = modCoordinates.attr("Extent2D");

    clsBox2D.attr("EPSILON") = py::float_(Box2D::EPSILON);
    clsBox2D.attr("INVALID") = py::float_(Box2D::INVALID);

    clsBox2D.def(py::init<>());
    clsBox2D.def(py::init<Point2D const &, Point2D const &, bool>(),
        "minimum"_a, "maximum"_a, "invert"_a=true);
    clsBox2D.def(py::init<Point2D const &, Extent2D const &, bool>(),
        "minimum"_a, "dimensions"_a, "invert"_a=true);
    clsBox2D.def(py::init<Box2I const &>());
    clsBox2D.def(py::init<Box2D const &>());

    clsBox2D.def("__eq__",
                 [](Box2D const & self, Box2D const & other) { return self == other; },
                 py::is_operator());
    clsBox2D.def("__ne__",
                 [](Box2D const & self, Box2D const & other) { return self != other; },
                 py::is_operator());

    clsBox2D.def("swap", &Box2D::swap);
    clsBox2D.def("getMin", &Box2D::getMin);
    clsBox2D.def("getMinX", &Box2D::getMinX);
    clsBox2D.def("getMinY", &Box2D::getMinY);
    clsBox2D.def("getMax", &Box2D::getMax);
    clsBox2D.def("getMaxX", &Box2D::getMaxX);
    clsBox2D.def("getMaxY", &Box2D::getMaxY);
    clsBox2D.def("getDimensions", &Box2D::getDimensions);
    clsBox2D.def("getWidth", &Box2D::getWidth);
    clsBox2D.def("getHeight", &Box2D::getHeight);
    clsBox2D.def("getArea", &Box2D::getArea);
    clsBox2D.def("getCenter", &Box2D::getCenter);
    clsBox2D.def("getCenterX", &Box2D::getCenterX);
    clsBox2D.def("getCenterY", &Box2D::getCenterY);
    clsBox2D.def("isEmpty", &Box2D::isEmpty);
    clsBox2D.def("contains", (bool (Box2D::*)(Point2D const &) const) &Box2D::contains);
    clsBox2D.def("contains", (bool (Box2D::*)(Box2D const &) const) &Box2D::contains);
    clsBox2D.def("overlaps", &Box2D::overlaps);
    clsBox2D.def("grow", (void (Box2D::*)(double)) &Box2D::grow);
    clsBox2D.def("grow", (void (Box2D::*)(Extent2D const&)) &Box2D::grow);
    clsBox2D.def("overlaps", (void (Box2D::*)(Extent2D const &)) &Box2D::overlaps);
    clsBox2D.def("shift", &Box2D::shift);
    clsBox2D.def("flipLR", &Box2D::flipLR);
    clsBox2D.def("flipTB", &Box2D::flipTB);
    clsBox2D.def("include", (void (Box2D::*)(Point2D const &)) &Box2D::include);
    clsBox2D.def("include", (void (Box2D::*)(Box2D const &)) &Box2D::include);
    clsBox2D.def("clip", &Box2D::clip);
    clsBox2D.def("getCorners", &Box2D::getCorners);
    clsBox2D.def("toString", &Box2D::toString);
    clsBox2D.def(
        "__repr__", [](Box2D const & self) {
            return py::str("Box2D(minimum={}, dimensions={})").format(
                py::repr(py::cast(self.getMin())),
                py::repr(py::cast(self.getDimensions()))
            );
        }
    );
    clsBox2D.def(
        "__str__", [](Box2D const & self) {
            return py::str("(minimum={}, maximum={})").format(
                py::str(py::cast(self.getMin())),
                py::str(py::cast(self.getMax()))
            );
        }
    );
    clsBox2D.def(
        "__reduce__",
        [clsBox2D](Box2D const & self) {
            return py::make_tuple(
                clsBox2D,
                make_tuple(py::cast(self.getMin()), py::cast(self.getMax()))
            );
        }
    );

    /* module-level typedefs */
    mod.attr("BoxI") = clsBox2I;
    mod.attr("BoxD") = clsBox2D;

    return mod.ptr();
}

}}}} // namespace lsst::afw::geom::<anonymous>
