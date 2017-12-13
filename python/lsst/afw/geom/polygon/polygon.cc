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

#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/pex/exceptions/Runtime.h"
#include "lsst/pex/exceptions/python/Exception.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/AffineTransform.h"
#include "lsst/afw/geom/Transform.h"
#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/geom/polygon/Polygon.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace polygon {

PYBIND11_PLUGIN(_polygon) {
    py::module mod("_polygon", "Python wrapper for afw _polygon library");

    // TODO: Commented-out code is waiting until needed and is untested.
    // Add tests for it and enable it or remove it before the final pybind11 merge.

    /* Module level */
    py::class_<Polygon, std::shared_ptr<Polygon>> clsPolygon(mod, "Polygon");

    pex::exceptions::python::declareException<SinglePolygonException, pex::exceptions::RuntimeError>(
            mod, "SinglePolygonException", "RuntimeError");

    /* Member types and enums */

    /* Constructors */
    clsPolygon.def(py::init<Polygon::Box const &>());
    clsPolygon.def(py::init<Polygon::Box const &, TransformPoint2ToPoint2 const &>());
    clsPolygon.def(py::init<Polygon::Box const &, std::shared_ptr<XYTransform const> const &>());
    clsPolygon.def(py::init<Polygon::Box const &, AffineTransform const &>());
    clsPolygon.def(py::init<std::vector<Polygon::Point> const &>());

    table::io::python::addPersistableMethods<Polygon>(clsPolygon);

    /* Operators */
    clsPolygon.def("__eq__", [](Polygon const &self, Polygon const &other) { return self == other; },
                   py::is_operator());
    clsPolygon.def("__ne__", [](Polygon const &self, Polygon const &other) { return self != other; },
                   py::is_operator());

    /* Members */
    clsPolygon.def("getNumEdges", &Polygon::getNumEdges);
    clsPolygon.def("getBBox", &Polygon::getBBox);
    clsPolygon.def("calculateCenter", &Polygon::calculateCenter);
    clsPolygon.def("calculateArea", &Polygon::calculateArea);
    clsPolygon.def("calculatePerimeter", &Polygon::calculatePerimeter);
    clsPolygon.def("getVertices", &Polygon::getVertices);
    clsPolygon.def("getEdges", &Polygon::getEdges);
    clsPolygon.def("contains", &Polygon::contains);
    clsPolygon.def("overlaps", (bool (Polygon::*)(Polygon const &) const) & Polygon::overlaps);
    clsPolygon.def("overlaps", (bool (Polygon::*)(Polygon::Box const &) const) & Polygon::overlaps);
    clsPolygon.def("intersectionSingle", (std::shared_ptr<Polygon> (Polygon::*)(Polygon const &) const) &
                                                 Polygon::intersectionSingle);
    clsPolygon.def("intersectionSingle", (std::shared_ptr<Polygon> (Polygon::*)(Polygon::Box const &) const) &
                                                 Polygon::intersectionSingle);
    clsPolygon.def("intersection",
                   (std::vector<std::shared_ptr<Polygon>> (Polygon::*)(Polygon const &) const) &
                           Polygon::intersection);
    clsPolygon.def("intersection",
                   (std::vector<std::shared_ptr<Polygon>> (Polygon::*)(Polygon::Box const &) const) &
                           Polygon::intersection);
    clsPolygon.def("unionSingle",
                   (std::shared_ptr<Polygon> (Polygon::*)(Polygon const &) const) & Polygon::unionSingle);
    clsPolygon.def("unionSingle", (std::shared_ptr<Polygon> (Polygon::*)(Polygon::Box const &) const) &
                                          Polygon::unionSingle);

    // Wrap Polygon::union_ (C++) as Polygon.union (Python)
    clsPolygon.def("union", (std::vector<std::shared_ptr<Polygon>> (Polygon::*)(Polygon const &) const) &
                                    Polygon::union_);
    clsPolygon.def("union", (std::vector<std::shared_ptr<Polygon>> (Polygon::*)(Polygon::Box const &) const) &
                                    Polygon::union_);
    clsPolygon.def("symDifference",
                   (std::vector<std::shared_ptr<Polygon>> (Polygon::*)(Polygon const &) const) &
                           Polygon::symDifference);
    clsPolygon.def("symDifference",
                   (std::vector<std::shared_ptr<Polygon>> (Polygon::*)(Polygon::Box const &) const) &
                           Polygon::symDifference);
    // clsPolygon.def("simplify", &Polygon::simplify);
    clsPolygon.def("convexHull", &Polygon::convexHull);
    clsPolygon.def("transform",
                   (std::shared_ptr<Polygon> (Polygon::*)(TransformPoint2ToPoint2 const &) const) &
                           Polygon::transform);
    clsPolygon.def("transform",
                   (std::shared_ptr<Polygon> (Polygon::*)(std::shared_ptr<XYTransform const> const &) const) &
                           Polygon::transform);
    clsPolygon.def("transform", (std::shared_ptr<Polygon> (Polygon::*)(AffineTransform const &) const) &
                                        Polygon::transform);
    clsPolygon.def("subSample", (std::shared_ptr<Polygon> (Polygon::*)(size_t) const) & Polygon::subSample);
    clsPolygon.def("subSample", (std::shared_ptr<Polygon> (Polygon::*)(double) const) & Polygon::subSample);
    clsPolygon.def("createImage",
                   (std::shared_ptr<afw::image::Image<float>> (Polygon::*)(Box2I const &) const) &
                           Polygon::createImage);
    clsPolygon.def("createImage",
                   (std::shared_ptr<afw::image::Image<float>> (Polygon::*)(Extent2I const &) const) &
                           Polygon::createImage);
    // clsPolygon.def("isPersistable", &Polygon::isPersistable);

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::geom::polygon
