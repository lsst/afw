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

namespace lsst {
namespace afw {
namespace math {
namespace detail {

namespace {
template <typename OutImageT, typename InImageT>
void declareByType(py::module &mod) {
    mod.def("basicConvolve",
            (void (*)(OutImageT &, InImageT const &, lsst::afw::math::Kernel const &,
                      lsst::afw::math::ConvolutionControl const &))basicConvolve<OutImageT, InImageT>);
    mod.def("basicConvolve",
            (void (*)(OutImageT &, InImageT const &, lsst::afw::math::DeltaFunctionKernel const &,
                      lsst::afw::math::ConvolutionControl const &))basicConvolve<OutImageT, InImageT>);
    mod.def("basicConvolve",
            (void (*)(OutImageT &, InImageT const &, lsst::afw::math::LinearCombinationKernel const &,
                      lsst::afw::math::ConvolutionControl const &))basicConvolve<OutImageT, InImageT>);
    mod.def("basicConvolve",
            (void (*)(OutImageT &, InImageT const &, lsst::afw::math::SeparableKernel const &,
                      lsst::afw::math::ConvolutionControl const &))basicConvolve<OutImageT, InImageT>);
    mod.def("convolveWithBruteForce",
            (void (*)(
                    OutImageT &, InImageT const &, lsst::afw::math::Kernel const &,
                    lsst::afw::math::ConvolutionControl const &))convolveWithBruteForce<OutImageT, InImageT>);
}
template <typename PixelType1, typename PixelType2>
void declareAll(py::module &mod) {
    using M1 = image::MaskedImage<PixelType1, image::MaskPixel, image::VariancePixel>;
    using M2 = image::MaskedImage<PixelType2, image::MaskPixel, image::VariancePixel>;

    declareByType<image::Image<PixelType1>, image::Image<PixelType2>>(mod);
    declareByType<M1, M2>(mod);
}
}  // namespace

PYBIND11_PLUGIN(convolve) {
    py::module mod("convolve");

    declareAll<double, double>(mod);
    declareAll<double, float>(mod);
    declareAll<double, int>(mod);
    declareAll<double, std::uint16_t>(mod);
    declareAll<float, float>(mod);
    declareAll<float, int>(mod);
    declareAll<float, std::uint16_t>(mod);
    declareAll<int, int>(mod);
    declareAll<std::uint16_t, std::uint16_t>(mod);

    py::class_<KernelImagesForRegion, std::shared_ptr<KernelImagesForRegion>> clsKernelImagesForRegion(
            mod, "KernelImagesForRegion");

    py::enum_<KernelImagesForRegion::Location>(clsKernelImagesForRegion, "Location")
            .value("BOTTOM_LEFT", KernelImagesForRegion::Location::BOTTOM_LEFT)
            .value("BOTTOM_RIGHT", KernelImagesForRegion::Location::BOTTOM_RIGHT)
            .value("TOP_LEFT", KernelImagesForRegion::Location::TOP_LEFT)
            .value("TOP_RIGHT", KernelImagesForRegion::Location::TOP_RIGHT)
            .export_values();

    clsKernelImagesForRegion.def(
            py::init<KernelImagesForRegion::KernelConstPtr, lsst::geom::Box2I const &,
                     lsst::geom::Point2I const &, bool>(),
            "kernelPtr"_a, "bbox"_a, "xy0"_a, "doNormalize"_a);
    clsKernelImagesForRegion.def(
            py::init<KernelImagesForRegion::KernelConstPtr, lsst::geom::Box2I const &,
                     lsst::geom::Point2I const &, bool, KernelImagesForRegion::ImagePtr,
                     KernelImagesForRegion::ImagePtr, KernelImagesForRegion::ImagePtr,
                     KernelImagesForRegion::ImagePtr>(),
            "kernelPtr"_a, "bbox"_a, "xy0"_a, "doNormalize"_a, "bottomLeftImagePtr"_a,
            "bottomRightImagePtr"_a, "topLeftImagePtr"_a, "topRightImagePtr"_a);

    clsKernelImagesForRegion.def("getBBox", &KernelImagesForRegion::getBBox);
    clsKernelImagesForRegion.def("getXY0", &KernelImagesForRegion::getXY0);
    clsKernelImagesForRegion.def("getDoNormalize", &KernelImagesForRegion::getDoNormalize);
    clsKernelImagesForRegion.def("getImage", &KernelImagesForRegion::getImage);
    clsKernelImagesForRegion.def("getKernel", &KernelImagesForRegion::getKernel);
    clsKernelImagesForRegion.def("getPixelIndex", &KernelImagesForRegion::getPixelIndex);
    clsKernelImagesForRegion.def("computeNextRow", &KernelImagesForRegion::computeNextRow);
    clsKernelImagesForRegion.def_static("getMinInterpolationSize",
                                        KernelImagesForRegion::getMinInterpolationSize);

    py::class_<RowOfKernelImagesForRegion, std::shared_ptr<RowOfKernelImagesForRegion>>
            clsRowOfKernelImagesForRegion(mod, "RowOfKernelImagesForRegion");

    clsRowOfKernelImagesForRegion.def(py::init<int, int>(), "nx"_a, "ny"_a);

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
}
}
}
}  // lsst::afw::math::detail