/*
 * This file is part of afw.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
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
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <cstdint>

#include <pybind11/pybind11.h>
#include <lsst/utils/python.h>

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
void declareAll(lsst::utils::python::WrapperCollection &wrappers) {
    wrappers.wrap([](auto &mod) {
        mod.def("writeFitsImage",
                (void (*)(int, ImageT const &, geom::SkyWcs const *, char const *)) & writeBasicFits<ImageT>,
                "fd"_a, "data"_a, "wcs"_a = NULL, "title"_a = NULL);

        mod.def("writeFitsImage",
                (void (*)(std::string const &, ImageT const &, geom::SkyWcs const *, char const *)) &
                        writeBasicFits<ImageT>,
                "filename"_a, "data"_a, "wcs"_a = NULL, "title"_a = NULL);
    });
}

}  // namespace

PYBIND11_MODULE(_simpleFits, mod) {
    lsst::utils::python::WrapperCollection wrappers(mod, "lsst.afw.display");
    declareAll<image::Image<std::uint16_t>>(wrappers);
    declareAll<image::Image<std::uint64_t>>(wrappers);
    declareAll<image::Image<int>>(wrappers);
    declareAll<image::Image<float>>(wrappers);
    declareAll<image::Image<double>>(wrappers);
    declareAll<image::Mask<std::uint16_t>>(wrappers);
    declareAll<image::Mask<image::MaskPixel>>(wrappers);
    wrappers.finish();
}

}  // namespace display
}  // namespace afw
}  // namespace lsst