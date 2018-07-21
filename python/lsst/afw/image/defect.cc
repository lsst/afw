/*
 * LSST Data Management System
 * Copyright 2008-2017 AURA/LSST.
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

#include "lsst/afw/image/Defect.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyDefectBase = py::class_<DefectBase, std::shared_ptr<DefectBase>>;

PYBIND11_MODULE(defect, mod) {
    py::module::import("lsst.afw.geom");

    PyDefectBase cls(mod, "DefectBase");

    /* Constructors */
    cls.def(py::init<const lsst::geom::Box2I &>(), "bbox"_a);

    /* Members */
    cls.def("getBBox", &DefectBase::getBBox);
    cls.def("getX0", &DefectBase::getX0);
    cls.def("getX1", &DefectBase::getX1);
    cls.def("getY0", &DefectBase::getY0);
    cls.def("getY1", &DefectBase::getY1);
    cls.def("clip", &DefectBase::clip);
    cls.def("shift", (void (DefectBase::*)(int, int)) & DefectBase::shift, "dx"_a, "dy"_a);
    cls.def("shift", (void (DefectBase::*)(lsst::geom::Extent2I const &)) & DefectBase::shift, "d"_a);
}
}
}
}
}  // lsst::afw::image::<anonymous>
