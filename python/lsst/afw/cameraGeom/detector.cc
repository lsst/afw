/*
 * LSST Data Management System
 * See COPYRIGHT file at the top of the source tree.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lsst/afw/cameraGeom/CameraPoint.h"
#include "lsst/afw/cameraGeom/CameraSys.h"
#include "lsst/afw/cameraGeom/Orientation.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/Extent.h"
#include "lsst/afw/geom/Point.h"
#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/geom/TransformMap.h"
#include "lsst/afw/table/AmpInfo.h"
#include "lsst/afw/cameraGeom/Detector.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

PYBIND11_PLUGIN(_detector) {
    py::module mod("_detector", "Python wrapper for afw _detector library");

    /* Module level */
    py::class_<Detector, std::shared_ptr<Detector>> cls(mod, "Detector");

    /* Member types and enums */
    py::enum_<DetectorType>(mod, "DetectorType")
            .value("SCIENCE", DetectorType::SCIENCE)
            .value("FOCUS", DetectorType::FOCUS)
            .value("GUIDER", DetectorType::GUIDER)
            .value("WAVEFRONT", DetectorType::WAVEFRONT)
            .export_values();

    /* Constructors */
    cls.def(py::init<std::string const &, int, DetectorType, std::string const &, geom::Box2I const &,
                     table::AmpInfoCatalog const &, Orientation const &, geom::Extent2D const &,
                     CameraTransformMap::Transforms const &>(),
            "name"_a, "id"_a, "type"_a, "serial"_a, "bbox"_a, "ampInfoCatalog"_a, "orientation"_a,
            "pixelSize"_a, "transforms"_a);

    /* Operators */
    cls.def("__getitem__",
            (std::shared_ptr<table::AmpInfoRecord const> (Detector::*)(int) const) & Detector::_get, "i"_a);
    cls.def("__getitem__",
            (std::shared_ptr<table::AmpInfoRecord const> (Detector::*)(std::string const &) const) &
                    Detector::_get,
            "name"_a);
    cls.def("__len__", &Detector::size);

    /* Members */
    cls.def("getName", &Detector::getName);
    cls.def("getId", &Detector::getId);
    cls.def("getType", &Detector::getType);
    cls.def("getSerial", &Detector::getSerial);
    cls.def("getBBox", &Detector::getBBox);
    cls.def("getCorners",
            (std::vector<geom::Point2D> (Detector::*)(CameraSys const &) const) & Detector::getCorners,
            "cameraSys"_a);
    cls.def("getCorners",
            (std::vector<geom::Point2D> (Detector::*)(CameraSysPrefix const &) const) & Detector::getCorners,
            "cameraSysPrefix"_a);
    cls.def("getCenter", (CameraPoint (Detector::*)(CameraSys const &) const) & Detector::getCenter,
            "getCenter"_a);
    cls.def("getCenter", (CameraPoint (Detector::*)(CameraSysPrefix const &) const) & Detector::getCenter,
            "getCenterPrefix"_a);
    cls.def("getAmpInfoCatalog", &Detector::getAmpInfoCatalog);
    cls.def("getOrientation", &Detector::getOrientation);
    cls.def("getPixelSize", &Detector::getPixelSize);
    cls.def("getTransformMap", &Detector::getTransformMap);
    cls.def("hasTransform", (bool (Detector::*)(CameraSys const &) const) & Detector::hasTransform,
            "cameraSys"_a);
    cls.def("hasTransform", (bool (Detector::*)(CameraSysPrefix const &) const) & Detector::hasTransform,
            "cameraSysPrefix"_a);
    cls.def("getTransform",
            (std::shared_ptr<geom::XYTransform const> (Detector::*)(CameraSys const &) const) &
                    Detector::getTransform,
            "cameraSys"_a);
    cls.def("getTransform",
            (std::shared_ptr<geom::XYTransform const> (Detector::*)(CameraSysPrefix const &) const) &
                    Detector::getTransform,
            "cameraSysPrefix"_a);
    cls.def("makeCameraPoint",
            (CameraPoint (Detector::*)(geom::Point2D, CameraSys) const) & Detector::makeCameraPoint,
            "point"_a, "cameraSys"_a);
    cls.def("makeCameraPoint",
            (CameraPoint (Detector::*)(geom::Point2D, CameraSysPrefix) const) & Detector::makeCameraPoint,
            "point"_a, "cameraSysPrefix"_a);
    cls.def("makeCameraSys",
            (CameraSys const (Detector::*)(CameraSys const &) const) & Detector::makeCameraSys,
            "cameraSys"_a);
    cls.def("makeCameraSys",
            (CameraSys const (Detector::*)(CameraSysPrefix const &) const) & Detector::makeCameraSys,
            "cameraSysPrefix"_a);
    cls.def("transform",
            (CameraPoint (Detector::*)(CameraPoint const &, CameraSys const &) const) & Detector::transform,
            "fromCameraPoint"_a, "toSys"_a);
    cls.def("transform", (CameraPoint (Detector::*)(CameraPoint const &, CameraSysPrefix const &) const) &
                                 Detector::transform,
            "fromCameraPoint"_a, "toSys"_a);

    return mod.ptr();
}
}
}
}
