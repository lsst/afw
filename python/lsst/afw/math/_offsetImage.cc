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
static void declareOffsetImage(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrap([](auto &mod) {
        mod.def("offsetImage", offsetImage<ImageT>, "image"_a, "dx"_a, "dy"_a, "algorithmName"_a = "lanczos5",
                "buffer"_a = 0);
    });
}

template <typename ImageT>
static void declareRotateImageBy90(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrap(
            [](auto &mod) { mod.def("rotateImageBy90", rotateImageBy90<ImageT>, "image"_a, "nQuarter"_a); });
}

template <typename ImageT>
static void declareFlipImage(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrap(
            [](auto &mod) { mod.def("flipImage", flipImage<ImageT>, "inImage"_a, "flipLR"_a, "flipTB"_a); });
}

template <typename ImageT>
static void declareBinImage(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrap([](auto &mod) {
        mod.def("binImage",
                (std::shared_ptr<ImageT>(*)(ImageT const &, int const, int const,
                                            lsst::afw::math::Property const))binImage<ImageT>,
                "inImage"_a, "binX"_a, "binY"_a, "flags"_a = lsst::afw::math::MEAN);
        mod.def("binImage",
                (std::shared_ptr<ImageT>(*)(ImageT const &, int const,
                                            lsst::afw::math::Property const))binImage<ImageT>,
                "inImage"_a, "binsize"_a, "flags"_a = lsst::afw::math::MEAN);
    });
}
}  // namespace

void wrapOffsetImage(lsst::utils::python::WrapperCollection &wrappers) {
    using MaskPixel = lsst::afw::image::MaskPixel;
    wrappers.addSignatureDependency("lsst.afw.image");

    declareOffsetImage<lsst::afw::image::Image<int>>(wrappers);
    declareOffsetImage<lsst::afw::image::Image<float>>(wrappers);
    declareOffsetImage<lsst::afw::image::Image<double>>(wrappers);
    declareOffsetImage<lsst::afw::image::MaskedImage<int>>(wrappers);
    declareOffsetImage<lsst::afw::image::MaskedImage<float>>(wrappers);
    declareOffsetImage<lsst::afw::image::MaskedImage<double>>(wrappers);

    declareRotateImageBy90<lsst::afw::image::Image<std::uint16_t>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::Image<int>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::Image<float>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::Image<double>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<std::uint16_t>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<int>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<float>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::MaskedImage<double>>(wrappers);
    declareRotateImageBy90<lsst::afw::image::Mask<MaskPixel>>(wrappers);

    declareFlipImage<lsst::afw::image::Image<std::uint16_t>>(wrappers);
    declareFlipImage<lsst::afw::image::Image<int>>(wrappers);
    declareFlipImage<lsst::afw::image::Image<float>>(wrappers);
    declareFlipImage<lsst::afw::image::Image<double>>(wrappers);
    declareFlipImage<lsst::afw::image::MaskedImage<std::uint16_t>>(wrappers);
    declareFlipImage<lsst::afw::image::MaskedImage<int>>(wrappers);
    declareFlipImage<lsst::afw::image::MaskedImage<float>>(wrappers);
    declareFlipImage<lsst::afw::image::MaskedImage<double>>(wrappers);
    declareFlipImage<lsst::afw::image::Mask<MaskPixel>>(wrappers);

    declareBinImage<lsst::afw::image::Image<std::uint16_t>>(wrappers);
    declareBinImage<lsst::afw::image::Image<int>>(wrappers);
    declareBinImage<lsst::afw::image::Image<float>>(wrappers);
    declareBinImage<lsst::afw::image::Image<double>>(wrappers);
    declareBinImage<lsst::afw::image::MaskedImage<std::uint16_t>>(wrappers);
    declareBinImage<lsst::afw::image::MaskedImage<int>>(wrappers);
    declareBinImage<lsst::afw::image::MaskedImage<float>>(wrappers);
    declareBinImage<lsst::afw::image::MaskedImage<double>>(wrappers);
}
}  // namespace math
}  // namespace afw
}  // namespace lsst
