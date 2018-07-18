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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the LSST License Statement and
 * the GNU General Public License along with this program. If not,
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */
#include <ostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <type_traits>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "ndarray/pybind11.h"

#include "lsst/utils/python.h"

#include "lsst/geom/Point.h"
#include "lsst/geom/SpherePoint.h"
#include "lsst/afw/geom/Endpoint.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

/*
Add `__str__`, `__repr__` and `getClassPrefix` methods to an Endpoint pybind11 wrapper

str(self) = "GenericEndpoint(_nAxes_)" for GenericEndpoint, e.g. "GenericEndpoint(4)";
            "_typeName_()" for all other Endpoint classes, e.g. "SpherePointEndpoint()",
repr(self) = "lsst.afw.geom." + str(self), e.g. "lsst.afw.geom.GenericEndpoint(4)"
*/
template <typename PyClass>
void addStrAndRepr(PyClass& cls) {
    using Class = typename PyClass::type;  // C++ class associated with pybind11 wrapper class
    utils::python::addOutputOp(cls, "__str__");
    cls.def("__repr__", [](Class const& self) {
        std::ostringstream os;
        os << "lsst.afw.geom." << self;
        return os.str();
    });
    cls.def_static("getClassPrefix", &Class::getClassPrefix);
}

/*
Add getNPoints, dataFromPoint, dataFromArray, pointFromData and arrayFromData
*/
template <typename PyClass>
void addDataConverters(PyClass& cls) {
    using Class = typename PyClass::type;  // C++ class associated with pybind11 wrapper class
    cls.def("getNPoints", &Class::getNPoints);
    cls.def("dataFromPoint", &Class::dataFromPoint);
    cls.def("dataFromArray", &Class::dataFromArray);
    cls.def("arrayFromData", &Class::arrayFromData);
    cls.def("pointFromData", &Class::pointFromData);
}

/*
Add makeFrame method
*/
template <typename PyClass>
void addMakeFrame(PyClass& cls) {
    using Class = typename PyClass::type;  // C++ class associated with pybind11 wrapper class
    // return a deep copy so Python cannot modify the internal state
    cls.def("makeFrame", [](Class const& self) {
        auto frame = self.makeFrame();
        return frame->copy();
    });
}

// Allow Python classes to be compared across different BaseEndpoints
template <typename SelfClass, typename OtherClass, typename PyClass>
std::enable_if_t<std::is_base_of<SelfClass, OtherClass>::value> addEquals(PyClass& cls) {
    cls.def("__eq__", &SelfClass::operator==);
    cls.def("__ne__", &SelfClass::operator!=);
}

template <typename SelfClass, typename OtherClass, typename PyClass>
std::enable_if_t<!std::is_base_of<SelfClass, OtherClass>::value> addEquals(PyClass& cls) {
    cls.def("__eq__", [](SelfClass const& self, OtherClass const& other) { return false; });
    cls.def("__ne__", [](SelfClass const& self, OtherClass const& other) { return true; });
}

template <typename SelfClass, typename PyClass>
void addAllEquals(PyClass& cls) {
    addEquals<SelfClass, GenericEndpoint>(cls);
    addEquals<SelfClass, Point2Endpoint>(cls);
    addEquals<SelfClass, SpherePointEndpoint>(cls);
}

/*
 * Declare BaseVectorEndpoint<Point, Array>;
 * this is meant to be called by other `declare...` functions;
 */
template <typename Point, typename Array>
void declareBaseEndpoint(py::module& mod, std::string const& suffix) {
    using Class = BaseEndpoint<Point, Array>;
    std::string const pyClassName = "_BaseEndpoint" + suffix;

    py::class_<Class, std::shared_ptr<Class>> cls(mod, pyClassName.c_str());

    cls.def_property_readonly("nAxes", &Class::getNAxes);
    addDataConverters(cls);
    addMakeFrame(cls);
    cls.def("normalizeFrame", &Class::normalizeFrame);
    addAllEquals<Class>(cls);
}

// Declare BaseVectorEndpoint and all subclasses (the corresponding BaseEndpoint)
// This is meant to be called by other `declare...` functions;
template <typename Point>
void declareBaseVectorEndpoint(py::module& mod, std::string const& suffix) {
    using Class = BaseVectorEndpoint<Point>;
    using Array = typename Class::Array;
    std::string const pyClassName = "_BaseVectorEndpoint" + suffix;

    declareBaseEndpoint<Point, Array>(mod, suffix);

    py::class_<Class, std::shared_ptr<Class>, BaseEndpoint<Point, Array>> cls(mod, pyClassName.c_str());

    addDataConverters(cls);
}

// Declare GenericEndpoint and all subclasses
void declareGenericEndpoint(py::module& mod) {
    using Class = GenericEndpoint;
    using Point = typename Class::Point;
    using Array = typename Class::Array;

    declareBaseEndpoint<Point, Array>(mod, "Generic");

    py::class_<Class, std::shared_ptr<Class>, BaseEndpoint<Point, Array>> cls(mod, "GenericEndpoint");

    cls.def(py::init<int>(), "nAxes"_a);

    addStrAndRepr(cls);
}

/// @internal declare PointNEndpoint (for N = 2 or 3) and all subclasses
void declarePoint2Endpoint(py::module& mod) {
    using Class = Point2Endpoint;
    using Point = typename Class::Point;
    std::string const pointNumStr = "Point2";
    std::string const pyClassName = pointNumStr + "Endpoint";

    declareBaseVectorEndpoint<Point>(mod, pointNumStr);

    py::class_<Class, std::shared_ptr<Class>, BaseVectorEndpoint<Point>> cls(mod, pyClassName.c_str());

    cls.def(py::init<>());
    // do not wrap the constructor that takes nAxes; it is an implementation detail

    cls.def("normalizeFrame", &Class::normalizeFrame);
    addStrAndRepr(cls);
}

/// @internal declare SpherePointEndpoint and all subclasses
void declareSpherePointEndpoint(py::module& mod) {
    using Class = SpherePointEndpoint;
    using Point = typename Class::Point;

    declareBaseVectorEndpoint<Point>(mod, "SpherePoint");

    py::class_<Class, std::shared_ptr<Class>, BaseVectorEndpoint<Point>> cls(mod, "SpherePointEndpoint");

    cls.def(py::init<>());
    // do not wrap the constructor that takes nAxes; it is an implementation detail

    addMakeFrame(cls);
    cls.def("normalizeFrame", &Class::normalizeFrame);
    addStrAndRepr(cls);
}

PYBIND11_MODULE(endpoint, mod) {
    py::module::import("lsst.geom");

    declareGenericEndpoint(mod);
    declarePoint2Endpoint(mod);
    declareSpherePointEndpoint(mod);
}

}  // namespace
}  // namespace geom
}  // namespace afw
}  // namespace lsst
