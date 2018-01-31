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

#include "pybind11/pybind11.h"

#include "lsst/utils/python.h"

#include "lsst/afw/geom/Angle.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace geom {
namespace {

using PyAngle = py::class_<Angle>;
using PyAngleUnit = py::class_<AngleUnit>;

template <typename OtherT>
void declareAngleComparisonOperators(PyAngle& cls) {
    cls.def("__eq__", [](Angle const& self, OtherT const& other) { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](Angle const& self, OtherT const& other) { return self != other; },
            py::is_operator());
    cls.def("__le__", [](Angle const& self, OtherT const& other) { return self <= other; },
            py::is_operator());
    cls.def("__ge__", [](Angle const& self, OtherT const& other) { return self >= other; },
            py::is_operator());
    cls.def("__lt__", [](Angle const& self, OtherT const& other) { return self < other; }, py::is_operator());
    cls.def("__gt__", [](Angle const& self, OtherT const& other) { return self > other; }, py::is_operator());
}

PYBIND11_PLUGIN(angle) {
    py::module mod("angle");

    /* AngleUnit */

    PyAngleUnit clsAngleUnit(mod, "AngleUnit");

    clsAngleUnit.def("__eq__", [](AngleUnit const& self, AngleUnit const& other) { return self == other; },
                     py::is_operator());
    clsAngleUnit.def("__ne__", [](AngleUnit const& self, AngleUnit const& other) { return !(self == other); },
                     py::is_operator());
    clsAngleUnit.def("_mul", [](AngleUnit const& self, double other) { return other * self; },
                     py::is_operator());
    clsAngleUnit.def("_rmul", [](AngleUnit const& self, double other) { return other * self; },
                     py::is_operator());
    mod.attr("radians") = py::cast(radians);
    mod.attr("degrees") = py::cast(degrees);
    mod.attr("hours") = py::cast(hours);
    mod.attr("arcminutes") = py::cast(arcminutes);
    mod.attr("arcseconds") = py::cast(arcseconds);

    /* Angle */

    PyAngle clsAngle(mod, "Angle");

    clsAngle.def(py::init<double, AngleUnit>(), py::arg("val"), py::arg("units") = radians);
    clsAngle.def(py::init<>());

    declareAngleComparisonOperators<Angle>(clsAngle);
    declareAngleComparisonOperators<double>(clsAngle);
    declareAngleComparisonOperators<int>(clsAngle);

    clsAngle.def("__mul__", [](Angle const& self, double other) { return self * other; }, py::is_operator());
    clsAngle.def("__mul__", [](Angle const& self, int other) { return self * other; }, py::is_operator());
    clsAngle.def("__rmul__", [](Angle const& self, double other) { return self * other; }, py::is_operator());
    clsAngle.def("__rmul__", [](Angle const& self, int other) { return self * other; }, py::is_operator());
    clsAngle.def("__imul__", [](Angle& self, double other) { return self *= other; });
    clsAngle.def("__imul__", [](Angle& self, int other) { return self *= other; });
    clsAngle.def("__add__", [](Angle const& self, Angle const& other) { return self + other; },
                 py::is_operator());
    clsAngle.def("__sub__", [](Angle const& self, Angle const& other) { return self - other; },
                 py::is_operator());
    clsAngle.def("__neg__", [](Angle const& self) { return -self; }, py::is_operator());
    clsAngle.def("__iadd__", [](Angle& self, Angle const& other) { return self += other; });
    clsAngle.def("__isub__", [](Angle& self, Angle const& other) { return self -= other; });
    clsAngle.def("__truediv__", [](Angle const& self, double other) { return self / other; },
                 py::is_operator());
    // Without an explicit wrapper, Python lets Angle / Angle -> Angle
    clsAngle.def("__truediv__", [](Angle const& self, Angle const& other) {
        throw py::type_error("unsupported operand type(s) for /: 'Angle' and 'Angle'");
    });

    clsAngle.def("__float__", &Angle::operator double);
    clsAngle.def("__abs__", [](Angle const& self) { return std::abs(self.asRadians()) * radians; });

    clsAngle.def("__reduce__", [clsAngle](Angle const& self) {
        return py::make_tuple(clsAngle, py::make_tuple(py::cast(self.asRadians())));
    });

    utils::python::addOutputOp(clsAngle, "__str__");
    utils::python::addOutputOp(clsAngle, "__repr__");

    clsAngle.def("asAngularUnits", &Angle::asAngularUnits);
    clsAngle.def("asRadians", &Angle::asRadians);
    clsAngle.def("asDegrees", &Angle::asDegrees);
    clsAngle.def("asHours", &Angle::asHours);
    clsAngle.def("asArcminutes", &Angle::asArcminutes);
    clsAngle.def("asArcseconds", &Angle::asArcseconds);
    clsAngle.def("wrap", &Angle::wrap);
    clsAngle.def("wrapCtr", &Angle::wrapCtr);
    clsAngle.def("wrapNear", &Angle::wrapNear);
    clsAngle.def("separation", &Angle::separation);

    /* Non-members */

    mod.attr("PI") = py::float_(PI);
    mod.attr("TWOPI") = py::float_(TWOPI);
    mod.attr("HALFPI") = py::float_(HALFPI);
    mod.attr("ONE_OVER_PI") = py::float_(ONE_OVER_PI);
    mod.attr("SQRTPI") = py::float_(SQRTPI);
    mod.attr("INVSQRTPI") = py::float_(INVSQRTPI);
    mod.attr("ROOT2") = py::float_(ROOT2);

    mod.def("degToRad", degToRad);
    mod.def("radToDeg", radToDeg);
    mod.def("radToArcsec", radToArcsec);
    mod.def("radToMas", radToMas);
    mod.def("arcsecToRad", arcsecToRad);
    mod.def("masToRad", masToRad);
    mod.def("isAngle", isAngle<Angle>);
    mod.def("isAngle", isAngle<double>);

    return mod.ptr();
}
}
}
}
}  // lsst::afw::geom::<anonymous>
