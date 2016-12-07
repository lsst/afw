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

#include <pybind11/pybind11.h>
//#include <pybind11/operators.h>
//#include <pybind11/stl.h>

#include "lsst/afw/cameraGeom/Detector.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace cameraGeom {

PYBIND11_PLUGIN(_detector) {
    py::module mod("_detector", "Python wrapper for afw _detector library");

    /* Module level */
    py::class_<Detector> clsDetector(mod, "Detector");

    /* Member types and enums */
    py::enum_<DetectorType>(mod, "DetectorType")
        .value("SCIENCE", DetectorType::SCIENCE)
        .value("FOCUS", DetectorType::FOCUS)
        .value("GUIDER", DetectorType::GUIDER)
        .value("WAVEFRONT", DetectorType::WAVEFRONT)
        .export_values();

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}

}}}
