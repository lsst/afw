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

#include "lsst/afw/geom/XYTransform.h"
#include "lsst/afw/image/DistortedTanWcs.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace image {
namespace {

using PyDistortedTanWcs = py::class_<DistortedTanWcs, std::shared_ptr<DistortedTanWcs>, TanWcs>;

PYBIND11_PLUGIN(distortedTanWcs) {
    py::module mod("distortedTanWcs");

    py::module::import("lsst.afw.geom");
    py::module::import("lsst.afw.image.tanWcs");

    /* Module level */
    PyDistortedTanWcs cls(mod, "DistortedTanWcs");

    /* Constructors */
    cls.def(py::init<TanWcs const &, geom::XYTransform const &>(), "tanWcs"_a, "pixelsToTanPixels"_a);

    /* Members */
    cls.def("clone", &DistortedTanWcs::clone);
    cls.def("flipImage", &DistortedTanWcs::flipImage, "flipLR"_a, "flipTB"_a, "dimensions"_a);
    cls.def("rotateImageBy90", &DistortedTanWcs::rotateImageBy90, "nQuarter"_a, "dimensions"_a);
    cls.def("shiftReferencePixel", &DistortedTanWcs::shiftReferencePixel, "dx"_a, "dy"_a);
    cls.def("hasDistortion", &DistortedTanWcs::hasDistortion);
    cls.def("getTanWcs", &DistortedTanWcs::getTanWcs);
    cls.def("getPixelToTanPixel", &DistortedTanWcs::getPixelToTanPixel);

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::image::<anonymous>
