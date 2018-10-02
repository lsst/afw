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
namespace {

using PyTransformMap = py::class_<TransformMap, std::shared_ptr<TransformMap>>;
using PyTransformMapBuilder = py::class_<TransformMap::Builder, std::shared_ptr<TransformMap::Builder>>;

void declareTransformMapBuilder(PyTransformMap & parent) {
    PyTransformMapBuilder cls(parent, "Builder");
    cls.def(py::init<CameraSys const &>(), "reference"_a);
    // connect overloads are wrapped with lambdas so we can return the Python
    // self object directly instead of re-converting *this to Python.
    cls.def("connect",
            [](py::object self, CameraSys const & fromSys, CameraSys const & toSys,
               std::shared_ptr<geom::TransformPoint2ToPoint2 const> transform) {
                py::cast<TransformMap::Builder &>(self).connect(fromSys, toSys, std::move(transform));
                return self;
            },
            "fromSys"_a, "toSys"_a, "transform"_a);
    cls.def("connect",
            [](py::object self, CameraSys const & fromSys, TransformMap::Transforms const & transforms) {
                py::cast<TransformMap::Builder &>(self).connect(fromSys, transforms);
                return self;
            },
            "fromSys"_a, "transforms"_a);
    cls.def("connect",
            [](py::object self, TransformMap::Transforms const & transforms) {
                py::cast<TransformMap::Builder &>(self).connect(transforms);
                return self;
            },
            "transforms"_a);
    cls.def("build", &TransformMap::Builder::build);
}

void declareTransformMap(py::module & mod) {
    PyTransformMap cls(mod, "TransformMap");

    cls.def(
        py::init([](
            CameraSys const &reference,
            TransformMap::Transforms const & transforms
        ) {
            // An apparent pybind11 bug: it's usually happy to cast away constness, but won't do it here.
            return std::const_pointer_cast<TransformMap>(TransformMap::make(reference, transforms));
        }),
        "reference"_a, "transforms"_a
    );

    cls.def("__len__", &TransformMap::size);
    cls.def("__contains__", &TransformMap::contains);
    cls.def("__iter__", [](TransformMap const &self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>()); /* Essential: keep object alive while iterator exists */

    cls.def(
        "transform",
        py::overload_cast<lsst::geom::Point2D const &, CameraSys const &, CameraSys const &>(
            &TransformMap::transform,
            py::const_
        ),
        "point"_a, "fromSys"_a, "toSys"_a
    );
    cls.def(
        "transform",
        py::overload_cast<std::vector<lsst::geom::Point2D> const &, CameraSys const &, CameraSys const &>(
            &TransformMap::transform,
            py::const_
        ),
        "pointList"_a, "fromSys"_a, "toSys"_a
    );
    cls.def("getTransform", &TransformMap::getTransform, "fromSys"_a, "toSys"_a);

    declareTransformMapBuilder(cls);
}

PYBIND11_MODULE(transformMap, mod) {
    declareTransformMap(mod);
}

}  // anonymous
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
