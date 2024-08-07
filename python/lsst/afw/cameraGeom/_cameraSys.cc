/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <string>

#include <pybind11/pybind11.h>
#include <lsst/cpputils/python.h>

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
    cls.def(
            "__eq__", [](CppClass const &self, CppClass const &other) { return self == other; },
            py::is_operator());
    cls.def(
            "__ne__", [](CppClass const &self, CppClass const &other) { return self != other; },
            py::is_operator());
    cpputils::python::addOutputOp(cls, "__str__");
    cpputils::python::addOutputOp(cls, "__repr__");
    cpputils::python::addHash(cls);

    /* Methods */
    cls.def("getSysName", &CppClass::getSysName);
}
}  // namespace

void wrapCameraSys(lsst::cpputils::python::WrapperCollection &wrappers) {
    /* Module level */
    wrappers.wrapType(py::class_<CameraSysPrefix>(wrappers.module, "CameraSysPrefix"),
                      [](auto &mod, auto &cls) {
                          declareCommonSysMethods<CameraSysPrefix>(cls);
                          cls.def(py::init<std::string const &>(), "sysName"_a);
                      });
    wrappers.wrapType(py::class_<CameraSys>(wrappers.module, "CameraSys"), [](auto &mod, auto &cls) {
        declareCommonSysMethods<CameraSys>(cls);
        /* Constructors */
        cls.def(py::init<std::string const &>(), "sysName"_a);
        cls.def(py::init<std::string const &, std::string const &>(), "sysName"_a, "detectorName"_a = "");
        cls.def(py::init<CameraSysPrefix const &, std::string const &>(), "sysPrefix"_a,
                "detectorName"_a = "");
        /* Members */
        cls.def("getDetectorName", &CameraSys::getDetectorName);
        cls.def("hasDetectorName", &CameraSys::hasDetectorName);
    });

    // The following must come after the associated pybind11 class is declared
    // (e.g. FOCAL_PLANE is a CameraSys, so clsCameraSys must have been declared
    wrappers.wrap([](auto &mod) {
        mod.attr("FOCAL_PLANE") = py::cast(FOCAL_PLANE);
        mod.attr("FIELD_ANGLE") = py::cast(FIELD_ANGLE);
        mod.attr("PIXELS") = py::cast(PIXELS);
        mod.attr("TAN_PIXELS") = py::cast(TAN_PIXELS);
        mod.attr("ACTUAL_PIXELS") = py::cast(ACTUAL_PIXELS);
    });
}
}  // namespace cameraGeom
}  // namespace afw
}  // namespace lsst
