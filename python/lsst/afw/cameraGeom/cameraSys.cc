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

#include "lsst/afw/cameraGeom/CameraSys.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace cameraGeom {

namespace {
    template <typename T, typename Class>
    void declareCommonSysMethods(Class & cls) {
        cls.def("getSysName", &T::getSysName);
    }
}

PYBIND11_PLUGIN(_cameraSys) {
    py::module mod("_cameraSys", "Python wrapper for afw _cameraSys library");

    /* Module level */
    py::class_<CameraSys> clsCameraSys(mod, "CameraSys");
    py::class_<CameraSysPrefix> clsCameraSysPrefix(mod, "CameraSysPrefix");

    // TODO: pybind11 only appropriate if we don't need any class members
    py::class_<CameraTransformMap> clsCameraTransformMap(mod, "CameraTransformMap");

    mod.attr("FOCAL_PLANE") = py::cast(FOCAL_PLANE);       // MUST come after clsCameraSys
    mod.attr("PUPIL") = py::cast(PUPIL);                   // MUST come after clsCameraSys
    mod.attr("ACTUAL_PIXELS") = py::cast(ACTUAL_PIXELS);   // MUST come after clsCameraSysPrefix
    mod.attr("PIXELS") = py::cast(PIXELS);                 // MUST come after clsCameraSysPrefix
    mod.attr("TAN_PIXELS") = py::cast(TAN_PIXELS);         // MUST come after clsCameraSysPrefix

    /* Member types and enums */
    declareCommonSysMethods<CameraSysPrefix>(clsCameraSysPrefix);
    declareCommonSysMethods<CameraSys>(clsCameraSys);

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}

}}}
