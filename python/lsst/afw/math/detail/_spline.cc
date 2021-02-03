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
#include <lsst/utils/python.h>

#include <pybind11/stl.h>

#include "lsst/afw/math/detail/Spline.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace lsst::afw::math::detail;
namespace lsst {
namespace afw {
namespace math {
namespace detail {
void wrapSpline(lsst::utils::python::WrapperCollection &wrappers) {
    /* Module level */
    wrappers.wrapType(py::class_<Spline>(wrappers.module, "Spline"), [](auto &mod, auto &cls) {
        cls.def("interpolate", &Spline::interpolate);
        cls.def("derivative", &Spline::derivative);
    });
    auto clsTautSpline = wrappers.wrapType(
            py::class_<TautSpline, Spline>(wrappers.module, "TautSpline"), [](auto &mod, auto &cls) {
                cls.def(py::init<std::vector<double> const &, std::vector<double> const &, double const,
                                 TautSpline::Symmetry>(),
                        "x"_a, "y"_a, "gamma"_a = 0, "type"_a = lsst::afw::math::detail::TautSpline::Unknown);
                cls.def("roots", &TautSpline::roots);
            });
    /* Member types and enums */
    wrappers.wrapType(py::enum_<TautSpline::Symmetry>(clsTautSpline, "Symmetry"), [](auto &mod, auto &enm) {
        enm.value("Unknown", TautSpline::Symmetry::Unknown);
        enm.value("Odd", TautSpline::Symmetry::Odd);
        enm.value("Even", TautSpline::Symmetry::Even);
        enm.export_values();
    });
}
}  // namespace detail
}  // namespace math
}  // namespace afw
}  // namespace lsst