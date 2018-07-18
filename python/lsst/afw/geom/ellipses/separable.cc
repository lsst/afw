/*
 * LSST Data Management System
 * Copyright 2008-2016  AURA/LSST.
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

#include "lsst/afw/geom/ellipses/Separable.h"
#include "lsst/afw/geom/ellipses/BaseCore.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/ConformalShear.h"
#include "lsst/afw/geom/ellipses/ReducedShear.h"
#include "lsst/afw/geom/ellipses/radii.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::geom::ellipses;

template <typename Ellipticity_, typename Radius_>
void declareSeparable(py::module &mod, const std::string &suffix) {
    using Class = Separable<Ellipticity_, Radius_>;

    py::class_<Class, std::shared_ptr<Class>, BaseCore> cls(mod, ("Separable" + suffix).c_str());

    //    py::enum_<typename Class::ParameterEnum>(mod, "ParameterEnum")
    //        .value("E0", Class::ParameterEnum::E0)
    //        .value("E1", Class::ParameterEnum::E1)
    //        .value("RADIUS", Class::ParameterEnum::RADIUS)
    //        .export_values();

    cls.def(py::init<double, double, double, bool>(), "e1"_a = 0.0, "e2"_a = 0.0, "radius"_a = Radius_(),
            "normalize"_a = true);
    //    cls.def(py::init<std::complex<double> const &, double, bool>(),
    //            "complex"_a, "radius"_a=Radius_(), "normalize"_a=true);
    //    cls.def(py::init<Ellipticity_ const &, double, bool>(),
    //            "ellipticity"_a, "radius"_a=Radius_(), "normalize"_a=true);
    //    cls.def(py::init<BaseCore::ParameterVector const &, bool>(),
    //            "vector"_a, "normalize"_a=true);
    cls.def(py::init<Class const &>());
    cls.def(py::init<BaseCore const &>());

    cls.def("getE1", &Class::getE1);
    cls.def("setE1", &Class::setE1);
    cls.def("getE2", &Class::getE2);
    cls.def("setE2", &Class::setE2);
    cls.def("getRadius", (Radius_ const &(Class::*)() const) & Class::getRadius);
    cls.def("setRadius", (void (Class::*)(double)) & Class::setRadius);
    cls.def("setRadius", (void (Class::*)(Radius_ const &)) & Class::setRadius);
    cls.def("getEllipticity", (Ellipticity_ const &(Class::*)() const) & Class::getEllipticity);
    cls.def("clone", &Class::clone);
    cls.def("getName", &Class::getName);
    cls.def("normalize", &Class::normalize);
    cls.def("assign", [](Class &self, Class &other) { self = other; });
    cls.def("assign", [](Class &self, BaseCore &other) { self = other; });
    cls.def("transform", [](Class &self, lsst::geom::LinearTransform const &t) {
        return std::static_pointer_cast<Class>(self.transform(t).copy());
    });
    cls.def("transformInPlace",
            [](Class &self, lsst::geom::LinearTransform const &t) { self.transform(t).inPlace(); });
    cls.def("__str__",
            [](Class &self) { return py::str("(%s, %s)").format(self.getEllipticity(), self.getRadius()); });
    cls.def("__repr__", [](Class &self) {
        return py::str("Separable(%r, %r)").format(self.getEllipticity(), self.getRadius());
    });
}

PYBIND11_PLUGIN(separable) {
    py::module mod("separable");

    declareSeparable<Distortion, DeterminantRadius>(mod, "DistortionDeterminantRadius");
    declareSeparable<Distortion, TraceRadius>(mod, "DistortionTraceRadius");
    declareSeparable<Distortion, LogDeterminantRadius>(mod, "DistortionLogDeterminantRadius");
    declareSeparable<Distortion, LogTraceRadius>(mod, "DistortionLogTraceRadius");

    declareSeparable<ConformalShear, DeterminantRadius>(mod, "ConformalShearDeterminantRadius");
    declareSeparable<ConformalShear, TraceRadius>(mod, "ConformalShearTraceRadius");
    declareSeparable<ConformalShear, LogDeterminantRadius>(mod, "ConformalShearLogDeterminantRadius");
    declareSeparable<ConformalShear, LogTraceRadius>(mod, "ConformalShearLogTraceRadius");

    declareSeparable<ReducedShear, DeterminantRadius>(mod, "ReducedShearDeterminantRadius");
    declareSeparable<ReducedShear, TraceRadius>(mod, "ReducedShearTraceRadius");
    declareSeparable<ReducedShear, LogDeterminantRadius>(mod, "ReducedShearLogDeterminantRadius");
    declareSeparable<ReducedShear, LogTraceRadius>(mod, "ReducedShearLogTraceRadius");

    return mod.ptr();
}