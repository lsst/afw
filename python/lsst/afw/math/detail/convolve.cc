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

#include "lsst/afw/math/detail/Convolve.h"

namespace py = pybind11;

using namespace py::literals;

using namespace lsst::afw::math::detail;

PYBIND11_PLUGIN(_convolve) {
    py::module mod("_convolve", "Python wrapper for afw _convolve library");

    py::class_<KernelImagesForRegion, std::shared_ptr<KernelImagesForRegion>> clsKernelImagesForRegion(mod, "KernelImagesForRegion");

    py::enum_<KernelImagesForRegion::Location>(clsKernelImagesForRegion, "Location")
        .value("BOTTOM_LEFT", KernelImagesForRegion::Location::BOTTOM_LEFT)
        .value("BOTTOM_RIGHT", KernelImagesForRegion::Location::BOTTOM_RIGHT)
        .value("TOP_LEFT", KernelImagesForRegion::Location::TOP_LEFT)
        .value("TOP_RIGHT", KernelImagesForRegion::Location::TOP_RIGHT)
        .export_values();

    clsKernelImagesForRegion.def(py::init<KernelImagesForRegion::KernelConstPtr,
            lsst::afw::geom::Box2I const &,
            lsst::afw::geom::Point2I const &,
            bool>(),
            "kernelPtr"_a,
            "bbox"_a,
            "xy0"_a,
            "doNormalize"_a);
    clsKernelImagesForRegion.def(py::init<KernelImagesForRegion::KernelConstPtr,
            lsst::afw::geom::Box2I const &,
            lsst::afw::geom::Point2I const &,
            bool,
            KernelImagesForRegion::ImagePtr,
            KernelImagesForRegion::ImagePtr,
            KernelImagesForRegion::ImagePtr,
            KernelImagesForRegion::ImagePtr>(),
            "kernelPtr"_a,
            "bbox"_a,
            "xy0"_a,
            "doNormalize"_a,
            "bottomLeftImagePtr"_a,
            "bottomRightImagePtr"_a,
            "topLeftImagePtr"_a,
            "topRightImagePtr"_a);

    clsKernelImagesForRegion.def("getBBox", &KernelImagesForRegion::getBBox);
    clsKernelImagesForRegion.def("getXY0", &KernelImagesForRegion::getXY0);
    clsKernelImagesForRegion.def("getDoNormalize", &KernelImagesForRegion::getDoNormalize);
    clsKernelImagesForRegion.def("getImage", &KernelImagesForRegion::getImage);
    clsKernelImagesForRegion.def("getKernel", &KernelImagesForRegion::getKernel);
    clsKernelImagesForRegion.def("getPixelIndex", &KernelImagesForRegion::getPixelIndex);
    clsKernelImagesForRegion.def("computeNextRow", &KernelImagesForRegion::computeNextRow);
    clsKernelImagesForRegion.def_static("getMinInterpolationSize", KernelImagesForRegion::getMinInterpolationSize);

    py::class_<RowOfKernelImagesForRegion, std::shared_ptr<RowOfKernelImagesForRegion>> clsRowOfKernelImagesForRegion(mod, "RowOfKernelImagesForRegion");

    clsRowOfKernelImagesForRegion.def(py::init<int, int>(),
            "nx"_a,
            "ny"_a);

    clsRowOfKernelImagesForRegion.def("front", &RowOfKernelImagesForRegion::front);
    clsRowOfKernelImagesForRegion.def("back", &RowOfKernelImagesForRegion::back);
    clsRowOfKernelImagesForRegion.def("getNX", &RowOfKernelImagesForRegion::getNX);
    clsRowOfKernelImagesForRegion.def("getNY", &RowOfKernelImagesForRegion::getNY);
    clsRowOfKernelImagesForRegion.def("getYInd", &RowOfKernelImagesForRegion::getYInd);
    clsRowOfKernelImagesForRegion.def("getRegion", &RowOfKernelImagesForRegion::getRegion);
    clsRowOfKernelImagesForRegion.def("hasData", &RowOfKernelImagesForRegion::hasData);
    clsRowOfKernelImagesForRegion.def("isLastRow", &RowOfKernelImagesForRegion::isLastRow);
    clsRowOfKernelImagesForRegion.def("incrYInd", &RowOfKernelImagesForRegion::incrYInd);

    /* Module level */

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}