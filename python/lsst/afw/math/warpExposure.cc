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

#include "lsst/afw/image/Wcs.h"
#include "lsst/afw/math/warpExposure.h"

namespace py = pybind11;

using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {
template<typename DestPixelT, typename SrcPixelT>
void declareWarpFunctionsByType(py::module & mod) {
    using DestExposureT = lsst::afw::image::Exposure<DestPixelT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;
    using SrcExposureT = lsst::afw::image::Exposure<SrcPixelT, lsst::afw::image::MaskPixel, lsst::afw::image::VariancePixel>;

// TODO: pybind11, decide if SinglePixel needs to be wrapped?
// If so, remove this one and add the uncommented one
    mod.def("warpExposure", [](DestExposureT &destExposure, SrcExposureT const &srcExposure, WarpingControl const &control) { return warpExposure(destExposure, srcExposure, control); },
        "destExposure"_a,
        "srcExposure"_a,
        "control"_a);
//    mod.def("warpExposure", warpExposure<DestExposureT, SrcExposureT>,
//        "destExposure"_a,
//        "srcExposure"_a,
//        "control"_a,
//        "padValue"_a=lsst::afw::math::edgePixel<typename DestExposureT::MaskedImageT>(typename lsst::afw::image::detail::image_traits<typename DestExposureT::MaskedImageT>::image_category()));

    mod.def("warpImage", (int (*)(lsst::afw::image::Image<DestPixelT> &, lsst::afw::image::Wcs const &, lsst::afw::image::Image<SrcPixelT> const &, lsst::afw::image::Wcs const &, WarpingControl const &, typename lsst::afw::image::Image<DestPixelT>::SinglePixel)) warpImage<lsst::afw::image::Image<DestPixelT>, lsst::afw::image::Image<SrcPixelT>>,
        "destImage"_a,
        "destWcs"_a,
        "srcImage"_a,
        "srcWcs"_a,
        "control"_a,
        "padValue"_a=lsst::afw::math::edgePixel<lsst::afw::image::Image<DestPixelT>>(typename lsst::afw::image::detail::image_traits<lsst::afw::image::Image<DestPixelT>>::image_category()));

    mod.def("warpImage", (int (*)(lsst::afw::image::Image<DestPixelT> &, lsst::afw::image::Image<SrcPixelT> const &, lsst::afw::geom::XYTransform const &, WarpingControl const &, typename lsst::afw::image::Image<DestPixelT>::SinglePixel)) warpImage<lsst::afw::image::Image<DestPixelT>, lsst::afw::image::Image<SrcPixelT>>,
        "destImage"_a,
        "srcImage"_a,
        "xyTransform"_a,
        "control"_a,
        "padValue"_a=lsst::afw::math::edgePixel<lsst::afw::image::Image<DestPixelT>>(typename lsst::afw::image::detail::image_traits<lsst::afw::image::Image<DestPixelT>>::image_category()));

    mod.def("warpCenteredImage", (int (*)(lsst::afw::image::Image<DestPixelT> &, lsst::afw::image::Image<SrcPixelT> const &, lsst::afw::geom::LinearTransform const &, lsst::afw::geom::Point2D const &, WarpingControl const &, typename lsst::afw::image::Image<DestPixelT>::SinglePixel)) warpCenteredImage<lsst::afw::image::Image<DestPixelT>>,
        "destImage"_a,
        "srcImage"_a,
        "linearTransform"_a,
        "centerPosition"_a,
        "control"_a,
        "padValue"_a=lsst::afw::math::edgePixel<lsst::afw::image::Image<DestPixelT>>(typename lsst::afw::image::detail::image_traits<lsst::afw::image::Image<DestPixelT>>::image_category()));
}
}

PYBIND11_PLUGIN(_warpExposure) {
    py::module mod("_warpExposure", "Python wrapper for afw _warpExposure library");

    py::class_<WarpingControl, std::shared_ptr<WarpingControl>> clsWarpingControl(mod, "WarpingControl");

    clsWarpingControl.def(py::init<std::string const &, std::string const &, int, int, lsst::afw::gpu::DevicePreference, lsst::afw::image::MaskPixel>(),
            "warpingKernelName"_a,
            "maskWarpingKernelName"_a="",
            "cacheSize"_a=0,
            "interpLength"_a=0,
            "devicePreference"_a=lsst::afw::gpu::DEFAULT_DEVICE_PREFERENCE,
            "growFullMask"_a=0);

    clsWarpingControl.def("getCacheSize", &WarpingControl::getCacheSize);
    clsWarpingControl.def("setCacheSize", &WarpingControl::setCacheSize);
    clsWarpingControl.def("getInterpLength", &WarpingControl::getInterpLength);
    clsWarpingControl.def("setInterpLength", &WarpingControl::setInterpLength);
    clsWarpingControl.def("getDevicePreference", &WarpingControl::getDevicePreference);
    clsWarpingControl.def("setDevicePreference", &WarpingControl::setDevicePreference);
    clsWarpingControl.def("getWarpingKernel", &WarpingControl::getWarpingKernel);
    clsWarpingControl.def("setWarpingKernelName", (void (WarpingControl::*)(std::string const &)) &WarpingControl::setWarpingKernelName);
    clsWarpingControl.def("setWarpingKernelName", (void (WarpingControl::*)(SeparableKernel const &)) &WarpingControl::setWarpingKernelName);
    clsWarpingControl.def("getMaskWarpingKernel", &WarpingControl::getMaskWarpingKernel);
    clsWarpingControl.def("hasMaskWarpingKernel", &WarpingControl::hasMaskWarpingKernel);
    clsWarpingControl.def("setMaskWarpingKernelName", (void (WarpingControl::*)(std::string const &)) &WarpingControl::setMaskWarpingKernelName);
    clsWarpingControl.def("setMaskWarpingKernelName", (void (WarpingControl::*)(SeparableKernel const &)) &WarpingControl::setMaskWarpingKernelName);
    clsWarpingControl.def("getGrowFullMask", &WarpingControl::getGrowFullMask);
    clsWarpingControl.def("setGrowFullMask", &WarpingControl::setGrowFullMask);

    /* Module level */
    declareWarpFunctionsByType<float, std::uint16_t>(mod);
    declareWarpFunctionsByType<double, std::uint16_t>(mod);
    declareWarpFunctionsByType<float, int>(mod);
    declareWarpFunctionsByType<double, int>(mod);
    declareWarpFunctionsByType<float, float>(mod);
    declareWarpFunctionsByType<double, float>(mod);
    declareWarpFunctionsByType<double, double>(mod);

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    return mod.ptr();
}
}}} // lsst::afw::math