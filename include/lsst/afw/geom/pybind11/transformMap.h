#ifndef AFW_GEOM_PYBIND11_TRANSFORMMAP_H_INCLUDED
#define AFW_GEOM_PYBIND11_TRANSFORMMAP_H_INCLUDED
/*
 * LSST Data Management System
 * Copyright 2016  AURA/LSST.
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
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/TransformMap.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace pybind11 {

/**
Declare an instantiation of TransformMap

@param[in] mod  pybind11 module.
@param[in] prefix  Prefix for python class name; full name = prefix + "TransformMap"
*/
template <typename CoordSysT>
py::class_<TransformMap<CoordSysT>, std::shared_ptr<TransformMap<CoordSysT>>>
    declareTransformMap(py::module & mod, std::string const & prefix)
{
    using Map = TransformMap<CoordSysT>;

    const std::string className = prefix + "TransformMap";
    py::class_<Map, std::shared_ptr<Map>> cls(mod, className.c_str());

    /* Constructors */
    cls.def(py::init<>());
    cls.def(py::init<CoordSysT const &, typename Map::Transforms const &>(),
            "nativeCoordSys"_a, "transforms"_a);

    /* Operators */
    cls.def("__getitem__", &Map::operator[], "coordSys"_a);
    cls.def("__len__", &Map::size);
    cls.def("__contains__", &Map::contains);

    /* Members */
    cls.def("transform",
            (Point2D (Map::*)(Point2D const &, CoordSysT const &, CoordSysT const &) const)
                &Map::transform,
            "point"_a, "fromCoordSys"_a, "toCoordSys"_a);
    cls.def("transform",
            (std::vector<Point2D> (Map::*)(std::vector<Point2D> const &, CoordSysT const &, CoordSysT const &) const)
                &Map::transform,
            "points"_a, "fromCoordSys"_a, "toCoordSys"_a);
    cls.def("getNativeCoordSys", &Map::getNativeCoordSys);
    cls.def("getCoordSysList", &Map::getCoordSysList);

    return cls;
};

}}}} // lsst::afw::geom::pybind11

#endif