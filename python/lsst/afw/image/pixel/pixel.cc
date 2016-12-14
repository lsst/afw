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
#include <string>

#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

#include "lsst/afw/image/MaskedImage.h"
#include "lsst/afw/image/Pixel.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace image {
namespace pixel {

namespace {
    /**
    Declare a SinglePixel for a MaskedImage

    (Note that SinglePixel for Image is just the pixel type)

    @tparam  Image plane type
    @param mod  pybind11 module
    @param[in] name  Name of Python class, e.g. "SinglePixelI" if PixelT is `int`.
    */
    template <typename PixelT>
    void declareSinglePixel(py::module & mod, std::string const & name) {
        mod.def("makeSinglePixel", &makeSinglePixel<PixelT, MaskPixel, VariancePixel>, "x"_a, "m"_a, "v"_a);

        py::class_<SinglePixel<PixelT, MaskPixel, VariancePixel>> cls(mod, name.c_str());

        cls.def(py::init<double, int, double const>(), "image"_a, "mask"_a=0, "variance"_a=0);
        cls.def(py::init<int, int, double const>(), "image"_a, "mask"_a=0, "variance"_a=0);
    }
}

PYBIND11_PLUGIN(_pixel) {
    py::module mod("_pixel", "Python wrapper for afw _pixel library");

    declareSinglePixel<float>(mod, "SinglePixelF");
    declareSinglePixel<double>(mod, "SinglePixelD");
    declareSinglePixel<int>(mod, "SinglePixelI");
    declareSinglePixel<std::uint16_t>(mod, "SinglePixelU");
    // TODO: fix or remove; presently this fail with complaint:
    // call to constructor of 'SinglePixel<unsigned long long, unsigned short, float>' is ambiguous
    // and it points out the "int" and "double" constructors
    //declareSinglePixel<std::uint64_t>(mod, "SinglePixelL");

    return mod.ptr();
}

}}}}  // namespace lsst::afw::image::pixel
