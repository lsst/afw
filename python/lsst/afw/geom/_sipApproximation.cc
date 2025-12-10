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

#include "pybind11/pybind11.h"
#include <lsst/cpputils/python.h>
#include "pybind11/stl.h"
#include "pybind11/eigen.h"

#include "ndarray/pybind11.h"

#include "lsst/afw/geom/SipApproximation.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace geom {
namespace {

using PySipApproximation = py::classh<SipApproximation>;

void declareSipApproximation(lsst::cpputils::python::WrapperCollection &wrappers) {
    wrappers.wrapType(PySipApproximation(wrappers.module, "SipApproximation"), [](auto &mod, auto &cls) {
        cls.def(py::init<SkyWcs const &, lsst::geom::Box2D const &,
                         lsst::geom::Extent2I const &, int,
                         std::optional<lsst::geom::Point2D> const &, double>(),
                "target"_a, "bbox"_a, "gridShape"_a, "order"_a = 5,
                "pixelOrigin"_a = std::nullopt, "svdThreshold"_a = -1);
        cls.def("getOrder", &SipApproximation::getOrder);
        cls.def("getA", py::overload_cast<int, int>(&SipApproximation::getA, py::const_), "p"_a, "q"_a);
        cls.def("getB", py::overload_cast<int, int>(&SipApproximation::getB, py::const_), "p"_a, "q"_a);
        cls.def("getA", py::overload_cast<>(&SipApproximation::getA, py::const_));
        cls.def("getB", py::overload_cast<>(&SipApproximation::getB, py::const_));
        cls.def("getBBox", &SipApproximation::getBBox);
        cls.def("getSkyOrigin", &SipApproximation::getSkyOrigin);
        cls.def("getPixelOrigin", &SipApproximation::getPixelOrigin);
        cls.def("getCdMatrix", &SipApproximation::getCdMatrix);
        cls.def("getWcs", &SipApproximation::getWcs);
        cls.def("computeDeltas", &SipApproximation::computeDeltas);
    });
}
}  // namespace
void wrapSipApproximation(lsst::cpputils::python::WrapperCollection &wrappers) {
    declareSipApproximation(wrappers);
}
}  // namespace geom
}  // namespace afw
}  // namespace lsst
