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

#include "lsst/afw/math/ConvolveImage.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {
template <typename OutImageT, typename InImageT, typename KernelT>
void declareConvolve(py::module & mod) {
    mod.def("convolve", (void (*)(OutImageT&, InImageT const&, KernelT const&, ConvolutionControl const&)) convolve<OutImageT, InImageT, KernelT>,
            "convolvedImage"_a, "inImage"_a, "kernel"_a, "convolutionControl"_a=ConvolutionControl());
    mod.def("convolve", (void (*)(OutImageT&, InImageT const&, KernelT const&, bool, bool)) convolve<OutImageT, InImageT, KernelT>,
            "convolvedImage"_a, "inImage"_a, "kernel"_a, "doNormalize"_a, "doCopyEdge"_a=false);
}

template <typename ImageType1, typename ImageType2>
void declareScaledPlus(py::module & mod) {
    mod.def("scaledPlus", (void (*)(ImageType1 &, double, ImageType2 const &, double, ImageType2 const &)) scaledPlus);
}

template <typename ImageType1, typename ImageType2>
void declareByType(py::module & mod) {
    declareConvolve<ImageType1, ImageType2, AnalyticKernel>(mod);
    declareConvolve<ImageType1, ImageType2, DeltaFunctionKernel>(mod);
    declareConvolve<ImageType1, ImageType2, FixedKernel>(mod);
    declareConvolve<ImageType1, ImageType2, LinearCombinationKernel>(mod);
    declareConvolve<ImageType1, ImageType2, SeparableKernel>(mod);
    declareConvolve<ImageType1, ImageType2, Kernel>(mod);
    declareScaledPlus<ImageType1, ImageType2>(mod);
}

template <typename PixelType1, typename PixelType2>
void declareAll(py::module & mod) {
    using M1 = image::MaskedImage<PixelType1, image::MaskPixel, image::VariancePixel>;
    using M2 = image::MaskedImage<PixelType2, image::MaskPixel, image::VariancePixel>;

    declareByType<image::Image<PixelType1>, image::Image<PixelType2>>(mod);
    declareByType<M1, M2>(mod);
}
}

PYBIND11_PLUGIN(_convolveImage) {
    py::module mod("_convolveImage", "Python wrapper for afw _convolveImage library");

    py::class_<ConvolutionControl, std::shared_ptr<ConvolutionControl>> clsConvolutionControl(mod, "ConvolutionControl");

    clsConvolutionControl.def(py::init<bool, bool, int>(),
            "doNormalize"_a=true, "doCopyEdge"_a=false, "maxInterpolationDistance"_a=10);

    clsConvolutionControl.def("getDoNormalize", &ConvolutionControl::getDoNormalize);
    clsConvolutionControl.def("getDoCopyEdge", &ConvolutionControl::getDoCopyEdge);
    clsConvolutionControl.def("getMaxInterpolationDistance", &ConvolutionControl::getMaxInterpolationDistance);
    clsConvolutionControl.def("setDoNormalize", &ConvolutionControl::setDoNormalize);
    clsConvolutionControl.def("setDoCopyEdge", &ConvolutionControl::setDoCopyEdge);
    clsConvolutionControl.def("setMaxInterpolationDistance", &ConvolutionControl::setMaxInterpolationDistance);

    declareAll<double, double>(mod);
    declareAll<double, float>(mod);
    declareAll<double, int>(mod);
    declareAll<double, std::uint16_t>(mod);
    declareAll<float, float>(mod);
    declareAll<float, int>(mod);
    declareAll<float, std::uint16_t>(mod);
    declareAll<int, int>(mod);
    declareAll<std::uint16_t, std::uint16_t>(mod);

    return mod.ptr();
}
}}} // lsst::afw::math