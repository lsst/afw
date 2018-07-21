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
//#include <pybind11/stl.h>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "lsst/afw/math/offsetImage.h"

namespace py = pybind11;

using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {
template <typename ImageT>
static void declareOffsetImage(py::module& mod) {
    mod.def("offsetImage", offsetImage<ImageT>, "image"_a, "dx"_a, "dy"_a, "algorithmName"_a = "lanczos5",
            "buffer"_a = 0);
}

template <typename ImageT>
static void declareRotateImageBy90(py::module& mod) {
    mod.def("rotateImageBy90", rotateImageBy90<ImageT>, "image"_a, "nQuarter"_a);
}

template <typename ImageT>
static void declareFlipImage(py::module& mod) {
    mod.def("flipImage", flipImage<ImageT>, "inImage"_a, "flipLR"_a, "flipTB"_a);
}

template <typename ImageT>
static void declareBinImage(py::module& mod) {
    mod.def("binImage", (std::shared_ptr<ImageT>(*)(ImageT const&, int const, int const,
                                                    lsst::afw::math::Property const))binImage<ImageT>,
            "inImage"_a, "binX"_a, "binY"_a, "flags"_a = lsst::afw::math::MEAN);
    mod.def("binImage", (std::shared_ptr<ImageT>(*)(ImageT const&, int const,
                                                    lsst::afw::math::Property const))binImage<ImageT>,
            "inImage"_a, "binsize"_a, "flags"_a = lsst::afw::math::MEAN);
}
}  // namespace

PYBIND11_MODULE(offsetImage, mod) {
    using MaskPixel = lsst::afw::image::MaskPixel;

    /* Module level */
    declareOffsetImage<lsst::afw::image::Image<int>>(mod);
    declareOffsetImage<lsst::afw::image::Image<float>>(mod);
    declareOffsetImage<lsst::afw::image::Image<double>>(mod);
    declareOffsetImage<lsst::afw::image::MaskedImage<int>>(mod);
    declareOffsetImage<lsst::afw::image::MaskedImage<float>>(mod);
    declareOffsetImage<lsst::afw::image::MaskedImage<double>>(mod);

    declareRotateImageBy90<lsst::afw::image::Image<std::uint16_t>>(mod);
    declareRotateImageBy90<lsst::afw::image::Image<int>>(mod);
    declareRotateImageBy90<lsst::afw::image::Image<float>>(mod);
    declareRotateImageBy90<lsst::afw::image::Image<double>>(mod);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<std::uint16_t>>(mod);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<int>>(mod);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<float>>(mod);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<double>>(mod);
    declareRotateImageBy90<lsst::afw::image::Mask<MaskPixel>>(mod);

    declareFlipImage<lsst::afw::image::Image<std::uint16_t>>(mod);
    declareFlipImage<lsst::afw::image::Image<int>>(mod);
    declareFlipImage<lsst::afw::image::Image<float>>(mod);
    declareFlipImage<lsst::afw::image::Image<double>>(mod);
    declareFlipImage<lsst::afw::image::MaskedImage<std::uint16_t>>(mod);
    declareFlipImage<lsst::afw::image::MaskedImage<int>>(mod);
    declareFlipImage<lsst::afw::image::MaskedImage<float>>(mod);
    declareFlipImage<lsst::afw::image::MaskedImage<double>>(mod);
    declareFlipImage<lsst::afw::image::Mask<MaskPixel>>(mod);

    declareBinImage<lsst::afw::image::Image<std::uint16_t>>(mod);
    declareBinImage<lsst::afw::image::Image<int>>(mod);
    declareBinImage<lsst::afw::image::Image<float>>(mod);
    declareBinImage<lsst::afw::image::Image<double>>(mod);
    declareBinImage<lsst::afw::image::MaskedImage<std::uint16_t>>(mod);
    declareBinImage<lsst::afw::image::MaskedImage<int>>(mod);
    declareBinImage<lsst::afw::image::MaskedImage<float>>(mod);
    declareBinImage<lsst::afw::image::MaskedImage<double>>(mod);
}
}
}
}  // lsst::afw::math
