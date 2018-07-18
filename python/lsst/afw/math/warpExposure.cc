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

#include <cstdint>
#include <memory>
#include <string>

#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

#include "lsst/afw/geom/SkyWcs.h"
#include "lsst/afw/image/Exposure.h"
#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/math/Kernel.h"
#include "lsst/afw/math/warpExposure.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace math {

namespace {
/**
@internal Declare a warping kernel class with no constructor

@tparam KernelT  class of warping kernel, e.g. LanczosWarpingKernel
@param[in] mod  pybind11 module to which to add the kernel
@param[in] name  Python name for class, e.g. "LanczosWarpingKernel"
@param[in] addConstructor  If true then add a default constructor.
*/
template <typename KernelT>
py::class_<KernelT, std::shared_ptr<KernelT>, SeparableKernel> declareWarpingKernel(py::module &mod,
                                                                                    std::string const &name) {
    py::class_<KernelT, std::shared_ptr<KernelT>, SeparableKernel> cls(mod, name.c_str());

    cls.def("clone", &KernelT::clone);
    return cls;
}

/**
@internal Declare a warping kernel class with a defaut constructor

@tparam KernelT  class of warping kernel, e.g. LanczosWarpingKernel
@param[in] mod  pybind11 module to which to add the kernel
@param[in] name  Python name for class, e.g. "LanczosWarpingKernel"
@param[in] addConstructor  If true then add a default constructor.
*/
template <typename KernelT>
py::class_<KernelT, std::shared_ptr<KernelT>, SeparableKernel> declareSimpleWarpingKernel(
        py::module &mod, std::string const &name, bool addConstructor = true) {
    auto cls = declareWarpingKernel<KernelT>(mod, name);
    cls.def(py::init<>());
    return cls;
}

/**
@internal Declare wrappers for warpImage and warpCenteredImage
for a particular pair of image or masked image types

@tparam DestImageT  Desination image type, e.g. Image<int> or MaskedImage<float, MaskType, VarianceType>
@tparam SrcImageT  Source image type, e.g. Image<int> or MaskedImage<float, MaskType, VarianceType>
@param[in,out] mod  pybind11 module for which to declare the function wrappers
*/
template <typename DestImageT, typename SrcImageT>
void declareImageWarpingFunctions(py::module &mod) {
    auto const EdgePixel =
            edgePixel<DestImageT>(typename image::detail::image_traits<DestImageT>::image_category());
    mod.def("warpImage", (int (*)(DestImageT &, geom::SkyWcs const &, SrcImageT const &, geom::SkyWcs const &,
                                  WarpingControl const &, typename DestImageT::SinglePixel)) &
                                 warpImage<DestImageT, SrcImageT>,
            "destImage"_a, "destWcs"_a, "srcImage"_a, "srcWcs"_a, "control"_a, "padValue"_a = EdgePixel);

    mod.def("warpImage",
            (int (*)(DestImageT &, SrcImageT const &, geom::TransformPoint2ToPoint2 const &,
                     WarpingControl const &, typename DestImageT::SinglePixel)) &
                    warpImage<DestImageT, SrcImageT>,
            "destImage"_a, "srcImage"_a, "srcToDest"_a, "control"_a, "padValue"_a = EdgePixel);

    mod.def("warpCenteredImage", &warpCenteredImage<DestImageT, SrcImageT>, "destImage"_a, "srcImage"_a,
            "linearTransform"_a, "centerPoint"_a, "control"_a, "padValue"_a = EdgePixel);
}

/**
@internal Declare wrappers for warpExposure, warpImage and warpCenteredImage
for a particular pair of source and destination pixel types.

Declares both image and masked image variants of warpImage and warpCenteredImage.

@tparam DestPixelT  Desination pixel type, e.g. `int` or `float`
@tparam SrcPixelT  Source pixel type, e.g. `int` or `float`
@param[in,out] mod  pybind11 module for which to declare the function wrappers
*/
template <typename DestPixelT, typename SrcPixelT>
void declareWarpingFunctions(py::module &mod) {
    using DestExposureT = image::Exposure<DestPixelT, image::MaskPixel, image::VariancePixel>;
    using SrcExposureT = image::Exposure<SrcPixelT, image::MaskPixel, image::VariancePixel>;
    using DestImageT = image::Image<DestPixelT>;
    using SrcImageT = image::Image<SrcPixelT>;
    using DestMaskedImageT = image::MaskedImage<DestPixelT, image::MaskPixel, image::VariancePixel>;
    using SrcMaskedImageT = image::MaskedImage<SrcPixelT, image::MaskPixel, image::VariancePixel>;

    mod.def("warpExposure", &warpExposure<DestExposureT, SrcExposureT>, "destExposure"_a, "srcExposure"_a,
            "control"_a, "padValue"_a = edgePixel<DestMaskedImageT>(
                                 typename image::detail::image_traits<DestMaskedImageT>::image_category()));

    declareImageWarpingFunctions<DestImageT, SrcImageT>(mod);
    declareImageWarpingFunctions<DestMaskedImageT, SrcMaskedImageT>(mod);
}
}

PYBIND11_PLUGIN(warpExposure) {
    py::module mod("warpExposure");

    /* Module level */
    auto clsLanczosWarpingKernel = declareWarpingKernel<LanczosWarpingKernel>(mod, "LanczosWarpingKernel");
    declareSimpleWarpingKernel<BilinearWarpingKernel>(mod, "BilinearWarpingKernel");
    declareSimpleWarpingKernel<NearestWarpingKernel>(mod, "NearestWarpingKernel");

    py::class_<WarpingControl, std::shared_ptr<WarpingControl>> clsWarpingControl(mod, "WarpingControl");

    declareWarpingFunctions<double, double>(mod);
    declareWarpingFunctions<double, float>(mod);
    declareWarpingFunctions<double, int>(mod);
    declareWarpingFunctions<double, std::uint16_t>(mod);
    declareWarpingFunctions<float, float>(mod);
    declareWarpingFunctions<float, int>(mod);
    declareWarpingFunctions<float, std::uint16_t>(mod);
    declareWarpingFunctions<int, int>(mod);
    declareWarpingFunctions<std::uint16_t, std::uint16_t>(mod);

    /* Member types and enums */

    /* Constructors */
    clsLanczosWarpingKernel.def(py::init<int>(), "order"_a);

    clsWarpingControl.def(py::init<std::string, std::string, int, int, image::MaskPixel>(),
                          "warpingKernelName"_a, "maskWarpingKernelName"_a = "", "cacheSize"_a = 0,
                          "interpLength"_a = 0, "growFullMask"_a = 0);

    /* Operators */
    clsLanczosWarpingKernel.def("getOrder", &LanczosWarpingKernel::getOrder);

    clsWarpingControl.def("getCacheSize", &WarpingControl::getCacheSize);
    clsWarpingControl.def("setCacheSize", &WarpingControl::setCacheSize, "cacheSize"_a);
    clsWarpingControl.def("getInterpLength", &WarpingControl::getInterpLength);
    clsWarpingControl.def("setInterpLength", &WarpingControl::setInterpLength, "interpLength"_a);
    clsWarpingControl.def("setWarpingKernelName", &WarpingControl::setWarpingKernelName,
                          "warpingKernelName"_a);
    clsWarpingControl.def("getWarpingKernel", &WarpingControl::getWarpingKernel);
    clsWarpingControl.def("setWarpingKernel", &WarpingControl::setWarpingKernel, "warpingKernel"_a);
    clsWarpingControl.def("setMaskWarpingKernelName", &WarpingControl::setMaskWarpingKernelName,
                          "maskWarpingKernelName"_a);
    clsWarpingControl.def("getMaskWarpingKernel", &WarpingControl::getMaskWarpingKernel);
    clsWarpingControl.def("hasMaskWarpingKernel", &WarpingControl::hasMaskWarpingKernel);
    clsWarpingControl.def("setMaskWarpingKernelName", &WarpingControl::setMaskWarpingKernelName,
                          "maskWarpingKernelName"_a);
    clsWarpingControl.def("setMaskWarpingKernel", &WarpingControl::setMaskWarpingKernel,
                          "maskWarpingKernel"_a);
    clsWarpingControl.def("getGrowFullMask", &WarpingControl::getGrowFullMask);
    clsWarpingControl.def("setGrowFullMask", &WarpingControl::setGrowFullMask, "growFullMask"_a);

    /* Members */

    return mod.ptr();
}
}
}
}  // namespace lsst::afw::math
