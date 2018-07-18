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
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <vector>

#include "lsst/geom/Point.h"
#include "lsst/afw/cameraGeom/CameraSys.h"
#include "lsst/afw/cameraGeom/TransformMap.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

PYBIND11_PLUGIN(transformMap) {
    py::module mod("transformMap");

    /* Module level */
    py::class_<TransformMap, std::shared_ptr<TransformMap>> cls(mod, "TransformMap");

    /* Member types and enums */

    /* Constructors */
    cls.def(pybind11::init<
                    CameraSys const &,
                    std::unordered_map<CameraSys, std::shared_ptr<geom::TransformPoint2ToPoint2>> const &>(),
            "reference"_a, "transforms"_a);

    /* Operators */
    cls.def("__len__", &TransformMap::size);
    cls.def("__contains__", &TransformMap::contains);
    cls.def("__iter__", [](const TransformMap &self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>()); /* Essential: keep object alive while iterator exists */

    /* Members */
    cls.def("transform",
            (lsst::geom::Point2D(TransformMap::*)(lsst::geom::Point2D const &, CameraSys const &, CameraSys const &)
                     const) &
                    TransformMap::transform,
            "point"_a, "fromSys"_a, "toSys"_a);
    cls.def("transform",
            (std::vector<lsst::geom::Point2D>(TransformMap::*)(std::vector<lsst::geom::Point2D> const &,
                                                         CameraSys const &, CameraSys const &) const) &
                    TransformMap::transform,
            "pointList"_a, "fromSys"_a, "toSys"_a);
    cls.def("getTransform", &TransformMap::getTransform, "fromSys"_a, "toSys"_a);

    return mod.ptr();
}

}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
