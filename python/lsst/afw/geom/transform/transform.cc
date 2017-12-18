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
#include "pybind11/pybind11.h"

#include <memory>

#include "astshim.h"
#include "pybind11/stl.h"
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"

#include "lsst/afw/table/io/python.h"
#include "lsst/afw/geom/Endpoint.h"
#include "lsst/afw/geom/Transform.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

// Return a string consisting of "_pythonClassName_[_fromNAxes_->_toNAxes_]",
// for example "TransformGenericToPoint2[4->2]"
template <class Class>
std::string formatStr(Class const &self, std::string const &pyClassName) {
    std::ostringstream os;
    os << pyClassName;
    auto const frameSet = self.getFrameSet();
    os << "[" << frameSet->getNIn() << "->" << frameSet->getNOut() << "]";
    return os.str();
}

template <class FromEndpoint, class ToEndpoint, class NextToEndpoint, class PyClass>
void declareMethodTemplates(PyClass &cls) {
    using ThisTransform = Transform<FromEndpoint, ToEndpoint>;
    using NextTransform = Transform<ToEndpoint, NextToEndpoint>;
    using SeriesTransform = Transform<FromEndpoint, NextToEndpoint>;
    // Need Python-specific logic to give sensible errors for mismatched Transform types
    cls.def("_then",
            (std::shared_ptr<SeriesTransform>(ThisTransform::*)(NextTransform const &, bool) const) &
                    ThisTransform::template then<NextToEndpoint>,
            "next"_a, "simplify"_a = true);
}

// Declare Transform<FromEndpoint, ToEndpoint> using python class name Transform<X>To<Y>
// where <X> and <Y> are the prefix of the from endpoint and to endpoint class, respectively,
// for example TransformGenericToPoint2
template <class FromEndpoint, class ToEndpoint>
void declareTransform(py::module &mod) {
    using Class = Transform<FromEndpoint, ToEndpoint>;
    using ToPoint = typename ToEndpoint::Point;
    using ToArray = typename ToEndpoint::Array;
    using FromPoint = typename FromEndpoint::Point;
    using FromArray = typename FromEndpoint::Array;

    std::string const pyClassName = Class::getShortClassName();

    py::class_<Class, std::shared_ptr<Class>> cls(mod, pyClassName.c_str());

    cls.def(py::init<ast::FrameSet const &, bool>(), "frameSet"_a, "simplify"_a = true);
    cls.def(py::init<ast::Mapping const &, bool>(), "mapping"_a, "simplify"_a = true);

    cls.def_property_readonly("hasForward", &Class::hasForward);
    cls.def_property_readonly("hasInverse", &Class::hasInverse);
    cls.def_property_readonly("fromEndpoint", &Class::getFromEndpoint);
    cls.def_property_readonly("toEndpoint", &Class::getToEndpoint);

    // Return a copy of the contained FrameSet in order to assure changing the returned FrameSet
    // will not affect the contained FrameSet (since Python ignores constness)
    cls.def("getFrameSet", [](Class const &self) { return self.getFrameSet()->copy(); });

    cls.def("applyForward", (ToArray(Class::*)(FromArray const &) const) & Class::applyForward, "array"_a);
    cls.def("applyForward", (ToPoint(Class::*)(FromPoint const &) const) & Class::applyForward, "point"_a);
    cls.def("applyInverse", (FromArray(Class::*)(ToArray const &) const) & Class::applyInverse, "array"_a);
    cls.def("applyInverse", (FromPoint(Class::*)(ToPoint const &) const) & Class::applyInverse, "point"_a);
    cls.def("getInverse", &Class::getInverse);
    /* Need some extra handling of ndarray return type in Python to prevent dimensions
     * of length 1 from being deleted */
    cls.def("_getJacobian", &Class::getJacobian);
    // Do not wrap getShortClassName because it returns the name of the class;
    // use `<class>.__name__` or `type(<instance>).__name__` instead.
    // Do not wrap readStream or writeStream because C++ streams are not easy to wrap.
    cls.def_static("readString", &Class::readString);
    cls.def("writeString", &Class::writeString);

    declareMethodTemplates<FromEndpoint, ToEndpoint, GenericEndpoint>(cls);
    declareMethodTemplates<FromEndpoint, ToEndpoint, Point2Endpoint>(cls);
    declareMethodTemplates<FromEndpoint, ToEndpoint, IcrsCoordEndpoint>(cls);

    // str(self) = "<Python class name>[<nIn>-><nOut>]"
    cls.def("__str__", [pyClassName](Class const &self) { return formatStr(self, pyClassName); });
    // repr(self) = "lsst.afw.geom.<Python class name>[<nIn>-><nOut>]"
    cls.def("__repr__",
            [pyClassName](Class const &self) { return "lsst.afw.geom." + formatStr(self, pyClassName); });

    table::io::python::addPersistableMethods<Class>(cls);
}

PYBIND11_PLUGIN(transform) {
    py::module mod("transform");

    py::module::import("astshim");
    py::module::import("lsst.afw.geom.endpoint");

    // Need to import numpy for ndarray and eigen conversions
    if (_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    declareTransform<GenericEndpoint, GenericEndpoint>(mod);
    declareTransform<GenericEndpoint, Point2Endpoint>(mod);
    declareTransform<GenericEndpoint, IcrsCoordEndpoint>(mod);
    declareTransform<Point2Endpoint, GenericEndpoint>(mod);
    declareTransform<Point2Endpoint, Point2Endpoint>(mod);
    declareTransform<Point2Endpoint, IcrsCoordEndpoint>(mod);
    declareTransform<IcrsCoordEndpoint, GenericEndpoint>(mod);
    declareTransform<IcrsCoordEndpoint, Point2Endpoint>(mod);
    declareTransform<IcrsCoordEndpoint, IcrsCoordEndpoint>(mod);

    return mod.ptr();
}

}  // namespace
}  // namespace geom
}  // namespace afw
}  // namespace lsst
