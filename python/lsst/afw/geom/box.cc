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

using PyBox2I = py::class_<Box2I, std::shared_ptr<Box2I>>;
using PyBox2D = py::class_<Box2D, std::shared_ptr<Box2D>>;

template <typename Box, typename ...Args>
void wrapCommonBoxInterface(py::class_<Box, Args...> & cls) {
    cls.def("__eq__", [](Box const &self, Box const &other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](Box const &self, Box const &other) { return self != other; },
            py::is_operator());
    cls.def("swap", &Box::swap);
    cls.def("getMin", &Box::getMin);
    cls.def("getMinX", &Box::getMinX);
    cls.def("getMinY", &Box::getMinY);
    cls.def("getMax", &Box::getMax);
    cls.def("getMaxX", &Box::getMaxX);
    cls.def("getMaxY", &Box::getMaxY);
    cls.def("getDimensions", &Box::getDimensions);
    cls.def("getWidth", &Box::getWidth);
    cls.def("getHeight", &Box::getHeight);
    cls.def("getArea", &Box::getArea);
    cls.def("isEmpty", &Box::isEmpty);
    cls.def("contains", (bool (Box::*)(typename Box::Point const &) const) & Box::contains);
    cls.def("contains", (bool (Box::*)(Box const &) const) & Box::contains);
    cls.def("__contains__", (bool (Box::*)(typename Box::Point const &) const) & Box::contains);
    cls.def("__contains__", (bool (Box::*)(Box const &) const) & Box::contains);
    cls.def("overlaps", &Box::overlaps);
    cls.def("grow", (void (Box::*)(typename Box::Element)) & Box::grow);
    cls.def("grow", (void (Box::*)(typename Box::Extent const &)) & Box::grow);
    cls.def("shift", &Box::shift);
    cls.def("flipLR", &Box::flipLR);
    cls.def("flipTB", &Box::flipTB);
    cls.def("include", (void (Box::*)(typename Box::Point const &)) & Box::include);
    cls.def("include", (void (Box::*)(Box const &)) & Box::include);
    cls.def("clip", &Box::clip);
    cls.def("getCorners", &Box::getCorners);
    cls.def("toString", &Box::toString);
    cls.def("__str__", [](Box const &self) {
        return py::str("(minimum={}, maximum={})")
                .format(py::str(py::cast(self.getMin())), py::str(py::cast(self.getMax())));
    });
    cls.def("__reduce__", [cls](Box const &self) {
        return py::make_tuple(cls, make_tuple(py::cast(self.getMin()), py::cast(self.getMax())));
    });
}

PYBIND11_PLUGIN(box) {
    py::module mod("box");

    py::object modCoordinates = py::module::import("lsst.afw.geom.coordinates");

    /* Box2UI */

    PyBox2I clsBox2I(mod, "Box2I");

    clsBox2I.attr("Point") = modCoordinates.attr("Point2I");
    clsBox2I.attr("Extent") = modCoordinates.attr("Extent2I");

    py::enum_<Box2I::EdgeHandlingEnum>(clsBox2I, "EdgeHandlingEnum")
            .value("EXPAND", Box2I::EdgeHandlingEnum::EXPAND)
            .value("SHRINK", Box2I::EdgeHandlingEnum::SHRINK)
            .export_values();

    clsBox2I.def(py::init<>());
    clsBox2I.def(py::init<Point2I const &, Point2I const &, bool>(), "minimum"_a, "maximum"_a,
                 "invert"_a = true);
    clsBox2I.def(py::init<Point2I const &, Extent2I const &, bool>(), "minimum"_a, "dimensions"_a,
                 "invert"_a = true);
    clsBox2I.def(py::init<Box2D const &, Box2I::EdgeHandlingEnum>(), "other"_a,
                 "edgeHandling"_a = Box2I::EXPAND);
    clsBox2I.def(py::init<Box2I const &>(), "other"_a);

    clsBox2I.def("getBegin", &Box2I::getBegin);
    clsBox2I.def("getBeginX", &Box2I::getBeginX);
    clsBox2I.def("getBeginY", &Box2I::getBeginY);
    clsBox2I.def("getEnd", &Box2I::getEnd);
    clsBox2I.def("getEndX", &Box2I::getEndX);
    clsBox2I.def("getEndY", &Box2I::getEndY);

    clsBox2I.def("__repr__", [](Box2I const &self) {
        return py::str("Box2I(minimum={}, dimensions={})")
                .format(py::repr(py::cast(self.getMin())), py::repr(py::cast(self.getDimensions())));
    });

    clsBox2I.def("getSlices", [](Box2I const &self) {
        return py::make_tuple(py::slice(self.getBeginY(), self.getEndY(), 1),
                              py::slice(self.getBeginX(), self.getEndX(), 1));
    });

    wrapCommonBoxInterface(clsBox2I);

    /* Box2D */

    PyBox2D clsBox2D(mod, "Box2D");

    clsBox2D.attr("Point") = modCoordinates.attr("Point2D");
    clsBox2D.attr("Extent") = modCoordinates.attr("Extent2D");

    clsBox2D.attr("EPSILON") = py::float_(Box2D::EPSILON);
    clsBox2D.attr("INVALID") = py::float_(Box2D::INVALID);

    clsBox2D.def(py::init<>());
    clsBox2D.def(py::init<Point2D const &, Point2D const &, bool>(), "minimum"_a, "maximum"_a,
                 "invert"_a = true);
    clsBox2D.def(py::init<Point2D const &, Extent2D const &, bool>(), "minimum"_a, "dimensions"_a,
                 "invert"_a = true);
    clsBox2D.def(py::init<Box2I const &>());
    clsBox2D.def(py::init<Box2D const &>());

    clsBox2D.def("getCenter", &Box2D::getCenter);
    clsBox2D.def("getCenterX", &Box2D::getCenterX);
    clsBox2D.def("getCenterY", &Box2D::getCenterY);

    clsBox2D.def("__repr__", [](Box2D const &self) {
        return py::str("Box2D(minimum={}, dimensions={})")
                .format(py::repr(py::cast(self.getMin())), py::repr(py::cast(self.getDimensions())));
    });

    wrapCommonBoxInterface(clsBox2D);

    /* module-level typedefs */
    mod.attr("BoxI") = clsBox2I;
    mod.attr("BoxD") = clsBox2D;

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::geom::<anonymous>
