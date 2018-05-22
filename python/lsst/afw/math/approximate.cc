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
#include <pybind11/stl.h>

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Approximate.h"

namespace py = pybind11;

using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {
template <typename PixelT>
void declareApproximate(py::module &mod, std::string const &suffix) {
    using Class = Approximate<PixelT>;

    py::class_<Class, std::shared_ptr<Class>> cls(mod, ("Approximate" + suffix).c_str());

    cls.def("getImage", &Class::getImage, "orderX"_a = -1, "orderY"_a = -1);
    cls.def("getMaskedImage", &Class::getMaskedImage, "orderX"_a = -1, "orderY"_a = -1);

    mod.def("makeApproximate",
            (std::shared_ptr<Approximate<PixelT>>(*)(std::vector<double> const &, std::vector<double> const &,
                                                     image::MaskedImage<PixelT> const &, lsst::geom::Box2I const &,
                                                     ApproximateControl const &))makeApproximate<PixelT>,
            "x"_a, "y"_a, "im"_a, "bbox"_a, "ctrl"_a);
}
}

PYBIND11_PLUGIN(_approximate) {
    py::module mod("_approximate", "Python wrapper for afw _approximate library");

    py::class_<ApproximateControl, std::shared_ptr<ApproximateControl>> clsApproximateControl(
            mod, "ApproximateControl");

    py::enum_<ApproximateControl::Style>(clsApproximateControl, "Style")
            .value("UNKNOWN", ApproximateControl::Style::UNKNOWN)
            .value("CHEBYSHEV", ApproximateControl::Style::CHEBYSHEV)
            .value("NUM_STYLES", ApproximateControl::Style::NUM_STYLES)
            .export_values();

    clsApproximateControl.def(py::init<ApproximateControl::Style, int, int, bool>(), "style"_a, "orderX"_a,
                              "orderY"_a = -1, "weighting"_a = true);

    clsApproximateControl.def("getStyle", &ApproximateControl::getStyle);
    clsApproximateControl.def("setStyle", &ApproximateControl::setStyle);
    clsApproximateControl.def("getOrderX", &ApproximateControl::getOrderX);
    clsApproximateControl.def("setOrderX", &ApproximateControl::setOrderX);
    clsApproximateControl.def("getOrderY", &ApproximateControl::getOrderY);
    clsApproximateControl.def("setOrderY", &ApproximateControl::setOrderY);
    clsApproximateControl.def("getWeighting", &ApproximateControl::getWeighting);
    clsApproximateControl.def("setWeighting", &ApproximateControl::setWeighting);

    // Yes, really only float
    declareApproximate<float>(mod, "F");

    return mod.ptr();
}
}
}
}  // lsst::afw::math