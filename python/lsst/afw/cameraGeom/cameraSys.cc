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

#include <string>

#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

#include "lsst/utils/python.h"

#include "lsst/afw/cameraGeom/CameraSys.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace cameraGeom {

namespace {
/**
@internal Declare methods common to CameraSysPrefix and CameraSys

@tparam CppClass  C++ class; one of CameraSysPrefix or CameraSys
@tparam PyClass  pybind11 class corresponding to `CppClass`
*/
template <typename CppClass, typename PyClass>
void declareCommonSysMethods(PyClass &cls) {
    /* Operators */
    cls.def("__eq__", [](CppClass const &self, CppClass const &other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](CppClass const &self, CppClass const &other) { return self != other; },
            py::is_operator());
    utils::python::addOutputOp(cls, "__str__");
    utils::python::addOutputOp(cls, "__repr__");
    utils::python::addHash(cls);

    /* Methods */
    cls.def("getSysName", &CppClass::getSysName);
}
}

PYBIND11_PLUGIN(cameraSys) {
    py::module mod("cameraSys");

    /* Module level */
    py::class_<CameraSysPrefix> clsCameraSysPrefix(mod, "CameraSysPrefix");
    py::class_<CameraSys> clsCameraSys(mod, "CameraSys");

    // The following must come after the associated pybind11 class is declared
    // (e.g. FOCAL_PLANE is a CameraSys, so clsCameraSys must have been declared
    mod.attr("FOCAL_PLANE") = py::cast(FOCAL_PLANE);
    mod.attr("FIELD_ANGLE") = py::cast(FIELD_ANGLE);
    mod.attr("PIXELS") = py::cast(PIXELS);
    mod.attr("TAN_PIXELS") = py::cast(TAN_PIXELS);
    mod.attr("ACTUAL_PIXELS") = py::cast(ACTUAL_PIXELS);

    /* Member types and enums */
    declareCommonSysMethods<CameraSysPrefix>(clsCameraSysPrefix);
    declareCommonSysMethods<CameraSys>(clsCameraSys);

    /* Constructors */
    clsCameraSysPrefix.def(py::init<std::string const &>(), "sysName"_a);
    clsCameraSys.def(py::init<std::string const &, std::string const &>(), "sysName"_a,
                     "detectorName"_a = "");
    clsCameraSys.def(py::init<CameraSysPrefix const &, std::string const &>(), "sysPrefix"_a,
                     "detectorName"_a = "");

    /* Operators */

    /* Members */
    clsCameraSys.def("getDetectorName", &CameraSys::getDetectorName);
    clsCameraSys.def("hasDetectorName", &CameraSys::hasDetectorName);

    return mod.ptr();
}
}
}
}
