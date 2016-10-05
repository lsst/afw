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

using namespace lsst::afw::math;

template <typename ImageType1, typename ImageType2>
void declareConvolveByType(py::module & mod) {
    /* Members */
    // declarations for convolve overloads go here...
    mod.def("scaledPlus", (void (*)(ImageType1 &, double, ImageType2 const &, double, ImageType2 const &)) scaledPlus);
}

template <typename PixelType1, typename PixelType2>
void declareConvolve(py::module & mod) {
    using lsst::afw::image::Image;
    using lsst::afw::image::MaskedImage;
    using lsst::afw::image::MaskPixel;
    using lsst::afw::image::VariancePixel;

    using M1 = MaskedImage<PixelType1, MaskPixel, VariancePixel>;
    using M2 = MaskedImage<PixelType2, MaskPixel, VariancePixel>;

    declareConvolveByType<Image<PixelType1>, Image<PixelType2>>(mod);
    declareConvolveByType<M1, M2>(mod);
}

PYBIND11_PLUGIN(_convolveImage) {
    py::module mod("_convolveImage", "Python wrapper for afw _convolveImage library");

    declareConvolve<double, double>(mod);
    declareConvolve<double, float>(mod);
    declareConvolve<double, int>(mod);
    declareConvolve<double, std::uint16_t>(mod);
    declareConvolve<float, float>(mod);
    declareConvolve<float, int>(mod);
    declareConvolve<float, std::uint16_t>(mod);
    declareConvolve<int, int>(mod);
    declareConvolve<std::uint16_t, std::uint16_t>(mod);

    return mod.ptr();
}