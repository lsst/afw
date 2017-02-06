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

#include "lsst/afw/geom/Functor.h"
#include "lsst/afw/geom/SeparableXYTransform.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst { namespace afw { namespace geom { namespace {

using PySeparableXYTransform = py::class_<SeparableXYTransform, XYTransform>;

PYBIND11_PLUGIN(_separableXYTransform) {
    py::module mod("_separableXYTransform");

    py::module::import("lsst.afw.geom._Functor");
    py::module::import("lsst.afw.geom._XYTransform");

    PySeparableXYTransform clsSeparableXYTransform(mod, "SeparableXYTransform");

    /* Constructors */
    clsSeparableXYTransform.def(py::init<Functor const &, Functor const &>(), "xfunctor"_a, "yfunctor"_a);

    /* Members */
    clsSeparableXYTransform.def("clone", &SeparableXYTransform::clone);
    clsSeparableXYTransform.def("forwardTransform", &SeparableXYTransform::forwardTransform);
    clsSeparableXYTransform.def("reverseTransform", &SeparableXYTransform::reverseTransform);
    // These return const references, but the Functor classes are immutable, so it's okay
    // to return them by reference (and impossible to just let pybind11 copy them, because
    // that would require a call to clone()).
    clsSeparableXYTransform.def("getXfunctor", &SeparableXYTransform::getXfunctor,
                                py::return_value_policy::reference_internal);
    clsSeparableXYTransform.def("getYfunctor", &SeparableXYTransform::getYfunctor,
                                py::return_value_policy::reference_internal);

    return mod.ptr();
}

}}}} // namespace lsst::afw::geom::<anonymous>
