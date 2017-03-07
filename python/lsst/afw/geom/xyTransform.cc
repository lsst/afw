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
#include "pybind11/stl.h"

#include "lsst/afw/geom/XYTransform.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst { namespace afw { namespace geom { namespace {

using PyXYTransform = py::class_<XYTransform, std::shared_ptr<XYTransform>>;

using PyIdentityXYTransform =
    py::class_<IdentityXYTransform, std::shared_ptr<IdentityXYTransform>, XYTransform>;

using PyInvertedXYTransform =
    py::class_<InvertedXYTransform, std::shared_ptr<InvertedXYTransform>, XYTransform>;

using PyMultiXYTransform =
    py::class_<MultiXYTransform, std::shared_ptr<MultiXYTransform>, XYTransform>;

using PyAffineXYTransfrom =
    py::class_<AffineXYTransform, std::shared_ptr<AffineXYTransform>, XYTransform>;

using PyRadialXYTransform =
    py::class_<RadialXYTransform, std::shared_ptr<RadialXYTransform>, XYTransform>;

PYBIND11_PLUGIN(xyTransform) {
    py::module mod("xyTransform");

    py::module::import("lsst.afw.geom.coordinates");
    py::module::import("lsst.afw.geom.affineTransform");

    PyXYTransform clsXYTransform(mod, "XYTransform");
    clsXYTransform.def("clone", &XYTransform::clone);
    clsXYTransform.def("invert", &XYTransform::invert);
    clsXYTransform.def("forwardTransform", &XYTransform::forwardTransform);
    clsXYTransform.def("reverseTransform", &XYTransform::reverseTransform);
    clsXYTransform.def("linearizeForwardTransform", &XYTransform::linearizeForwardTransform);
    clsXYTransform.def("linearizeReverseTransform", &XYTransform::linearizeReverseTransform);

    PyIdentityXYTransform clsIdentityXYTransform(mod, "IdentityXYTransform");
    clsIdentityXYTransform.def(py::init<>());
    clsIdentityXYTransform.def("forwardTransform", &IdentityXYTransform::forwardTransform);
    clsIdentityXYTransform.def("reverseTransform", &IdentityXYTransform::reverseTransform);
    clsIdentityXYTransform.def("linearizeForwardTransform", &IdentityXYTransform::linearizeForwardTransform);
    clsIdentityXYTransform.def("linearizeReverseTransform", &IdentityXYTransform::linearizeReverseTransform);

    PyInvertedXYTransform clsInvertedXYTransform(mod, "InvertedXYTransform");
    clsInvertedXYTransform.def(py::init<std::shared_ptr<XYTransform const>>());
    clsInvertedXYTransform.def("invert", &InvertedXYTransform::invert);
    clsInvertedXYTransform.def("forwardTransform", &InvertedXYTransform::forwardTransform);
    clsInvertedXYTransform.def("reverseTransform", &InvertedXYTransform::reverseTransform);
    clsInvertedXYTransform.def("linearizeForwardTransform", &InvertedXYTransform::linearizeForwardTransform);
    clsInvertedXYTransform.def("linearizeReverseTransform", &InvertedXYTransform::linearizeReverseTransform);

    PyMultiXYTransform clsMultiXYTransform(mod, "MultiXYTransform");
    clsMultiXYTransform.def(py::init<MultiXYTransform::TransformList const &>());
    clsMultiXYTransform.def("clone", &MultiXYTransform::clone);
    clsMultiXYTransform.def("forwardTransform", &MultiXYTransform::forwardTransform);
    clsMultiXYTransform.def("reverseTransform", &MultiXYTransform::reverseTransform);
    clsMultiXYTransform.def("linearizeForwardTransform", &MultiXYTransform::linearizeForwardTransform);
    clsMultiXYTransform.def("linearizeReverseTransform", &MultiXYTransform::linearizeReverseTransform);
    clsMultiXYTransform.def("getTransformList", &MultiXYTransform::getTransformList);

    PyAffineXYTransfrom clsAffineXYTransform(mod, "AffineXYTransform");
    clsAffineXYTransform.def(py::init<AffineTransform const &>());

    clsAffineXYTransform.def("clone", &AffineXYTransform::clone);
    clsAffineXYTransform.def("forwardTransform", &AffineXYTransform::forwardTransform);
    clsAffineXYTransform.def("reverseTransform", &AffineXYTransform::reverseTransform);
    clsAffineXYTransform.def("linearizeForwardTransform", &AffineXYTransform::linearizeForwardTransform);
    clsAffineXYTransform.def("linearizeReverseTransform", &AffineXYTransform::linearizeReverseTransform);
    clsAffineXYTransform.def("getForwardTransform", &AffineXYTransform::getForwardTransform);
    clsAffineXYTransform.def("getReverseTransform", &AffineXYTransform::getReverseTransform);

    PyRadialXYTransform clsRadialXYTransform(mod, "RadialXYTransform");
    clsRadialXYTransform.def(py::init<std::vector<double> const &>());
    clsRadialXYTransform.def("clone", &RadialXYTransform::clone);
    clsRadialXYTransform.def("invert", &RadialXYTransform::invert);
    clsRadialXYTransform.def("forwardTransform", &RadialXYTransform::forwardTransform);
    clsRadialXYTransform.def("reverseTransform", &RadialXYTransform::reverseTransform);
    clsRadialXYTransform.def("linearizeForwardTransform", &RadialXYTransform::linearizeForwardTransform);
    clsRadialXYTransform.def("linearizeReverseTransform", &RadialXYTransform::linearizeReverseTransform);
    clsRadialXYTransform.def("getCoeffs", &RadialXYTransform::getCoeffs);
    clsRadialXYTransform.def_static("polyInvert", &RadialXYTransform::polyInvert);
    clsRadialXYTransform.def_static(
        "polyEval", (double (*)(std::vector<double> const &, double)) & RadialXYTransform::polyEval);
    clsRadialXYTransform.def_static(
        "polyEval", (Point2D(*)(std::vector<double> const &, Point2D const &)) & RadialXYTransform::polyEval);
    clsRadialXYTransform.def_static("polyEvalDeriv", &RadialXYTransform::polyEvalDeriv);
    clsRadialXYTransform.def_static("polyEvalJacobian", &RadialXYTransform::polyEvalJacobian);
    clsRadialXYTransform.def_static(
        "polyEvalInverse", (double (*)(std::vector<double> const &, std::vector<double> const &, double)) &
                               RadialXYTransform::polyEvalInverse);
    clsRadialXYTransform.def_static(
        "polyEvalInverse",
        (Point2D(*)(std::vector<double> const &, std::vector<double> const &, Point2D const &)) &
            RadialXYTransform::polyEvalInverse);
    clsRadialXYTransform.def_static("polyEvalInverseJacobian", &RadialXYTransform::polyEvalInverseJacobian);
    clsRadialXYTransform.def_static("makeAffineTransform", &RadialXYTransform::makeAffineTransform);

    return mod.ptr();
}

}}}} // namespace lsst::afw::geom::<anonymous>