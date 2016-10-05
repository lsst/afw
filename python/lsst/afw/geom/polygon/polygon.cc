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
//#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/geom/polygon/Polygon.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::geom::polygon;

PYBIND11_PLUGIN(_polygon) {
    py::module mod("_polygon", "Python wrapper for afw _polygon library");

    py::class_<Polygon, std::shared_ptr<Polygon>> clsPolygon(mod, "Polygon");

    /* Module level */

    /* Member types and enums */

    /* Constructors */
    clsPolygon.def(py::init<Polygon::Box const &>());
//    clsPolygon.def(py::init<Polygon::Box const &, CONST_PTR(XYTransform) const &>());
//    clsPolygon.def(py::init<Polygon::Box const &, AffineTransform const &>());
    clsPolygon.def(py::init<std::vector<Polygon::Point> const &>());

    /* Operators */

    /* Members */
//    clsPolygon.def("swap", &Polygon::swap);
//    clsPolygon.def("getNumEdges", &Polygon::getNumEdges);
//    clsPolygon.def("getBPolygon::Box", &Polygon::getBPolygon::Box);
//    clsPolygon.def("calculateCenter", &Polygon::calculateCenter);
//    clsPolygon.def("calculateArea", &Polygon::calculateArea);
//    clsPolygon.def("calculatePerimeter", &Polygon::calculatePerimeter);
//    clsPolygon.def("getVertices", &Polygon::getVertices);
//    clsPolygon.def("getEdges", &Polygon::getEdges);
//    clsPolygon.def("contains", &Polygon::contains);
//    clsPolygon.def("overlaps", (bool (Polygon::*)(Polygon const &) const) &Polygon::overlaps);
//    clsPolygon.def("overlaps", (bool (Polygon::*)(Polygon::Box const &) const) &Polygon::overlaps);
//    clsPolygon.def("intersectionSingle", (PTR(Polygon) (Polygon::*)(Polygon const &) const) &Polygon::intersectionSingle);
//    clsPolygon.def("intersectionSingle", (PTR(Polygon) (Polygon::*)(Polygon::Box const &) const) &Polygon::intersectionSingle);
//    clsPolygon.def("intersection", (std::vector<PTR(Polygon)> (Polygon::*)(Polygon const &) const) &Polygon::intersection);
//    clsPolygon.def("intersection", (std::vector<PTR(Polygon)> (Polygon::*)(Polygon::Box const &) const) &Polygon::intersection);
//    clsPolygon.def("unionSingle", (PTR(Polygon) (Polygon::*)(Polygon const &) const) &Polygon::unionSingle);
//    clsPolygon.def("unionSingle", (PTR(Polygon) (Polygon::*)(Polygon::Box const &) const) &Polygon::unionSingle);
//
//    clsPolygon.def("union_", (std::vector<PTR(Polygon)> (Polygon::*)(Polygon const &) const) &Polygon::union_);
//    clsPolygon.def("union_", (std::vector<PTR(Polygon)> (Polygon::*)(Polygon::Box const &) const) &Polygon::union_);
//    clsPolygon.def("symDifference_", (std::vector<PTR(Polygon)> (Polygon::*)(Polygon const &) const) &Polygon::symDifference_);
//    clsPolygon.def("symDifference_", (std::vector<PTR(Polygon)> (Polygon::*)(Polygon::Box const &) const) &Polygon::symDifference_);
//    clsPolygon.def("simplify", &Polygon::simplify);
//    clsPolygon.def("convexHull", &Polygon::convexHull);
//    clsPolygon.def("transform" (PTR(Polygon) (Polygon::*)(CONST_PTR(XYTransform) const &) const) &Polygon::transform);
//    clsPolygon.def("transform" (PTR(Polygon) (Polygon::*)(AffineTransform const &) const) &Polygon::transform);
//    clsPolygon.def("subSample", (PTR(Polygon) (Polygon::*)(size_t) const) &Polygon::subSample);
//    clsPolygon.def("subSample", (PTR(Polygon) (Polygon::*)(double) const) &Polygon::subSample);
//    clsPolygon.def("createImage", (PTR::afw::image::image<float>) (Polygon::*)(Polygon::Box2I const &) const) &Polygon::createImage);
//    clsPolygon.def("createImage", (PTR::afw::image::image<float>) (Polygon::*)(Extent2I const &) const) &Polygon::createImage);
//    clsPolygon.def("isPersistable", &Polygon::isPersistable);

    return mod.ptr();
}