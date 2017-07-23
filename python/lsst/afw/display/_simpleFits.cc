/*
 * LSST Data Management System
 * Copyright 2008-2017  AURA/LSST.
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

#include <pybind11/pybind11.h>

#include "lsst/afw/image/Image.h"
#include "lsst/afw/image/Mask.h"
#include "simpleFits.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace display {

namespace {

template <typename ImageT>
void declareAll(py::module &mod) {
    mod.def("writeFitsImage",
            (void (*)(int, ImageT const &, image::Wcs const *, char const *)) & writeBasicFits<ImageT>,
            "fd"_a, "data"_a, "wcs"_a = NULL, "title"_a = NULL);

    mod.def("writeFitsImage",
            (void (*)(std::string const &, ImageT const &, image::Wcs const *, char const *)) &
                    writeBasicFits<ImageT>,
            "filename"_a, "data"_a, "wcs"_a = NULL, "title"_a = NULL);
}

}  // <anonymous>

PYBIND11_PLUGIN(_simpleFits) {
    py::module mod("_simpleFits", "");

    declareAll<image::Image<std::uint16_t>>(mod);
    declareAll<image::Image<std::uint64_t>>(mod);
    declareAll<image::Image<int>>(mod);
    declareAll<image::Image<float>>(mod);
    declareAll<image::Image<double>>(mod);
    declareAll<image::Mask<std::uint16_t>>(mod);
    declareAll<image::Mask<image::MaskPixel>>(mod);

    return mod.ptr();
}

}  // display
}  // afw
}  // lsst