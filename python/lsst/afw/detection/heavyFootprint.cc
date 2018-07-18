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
#include <pybind11/stl.h>

#include "ndarray/pybind11.h"

#include "lsst/afw/detection/Footprint.h"
#include "lsst/afw/detection/FootprintCtrl.h"
#include "lsst/afw/detection/HeavyFootprint.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

namespace {
template <typename ImagePixelT, typename MaskPixelT = lsst::afw::image::MaskPixel,
          typename VariancePixelT = lsst::afw::image::VariancePixel>
void declareHeavyFootprint(py::module &mod, std::string const &suffix) {
    using Class = HeavyFootprint<ImagePixelT>;
    py::class_<Class, std::shared_ptr<Class>, Footprint> clsHeavyFootprint(
            mod, ("HeavyFootprint" + suffix).c_str());

    /* Constructors */
    clsHeavyFootprint.def(
            py::init<Footprint const &,
                     lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> const &,
                     HeavyFootprintCtrl const *>(),
            "foot"_a, "mimage"_a, "ctrl"_a = nullptr);
    clsHeavyFootprint.def(py::init<Footprint const &, HeavyFootprintCtrl const *>(), "foot"_a,
                          "ctrl"_a = nullptr);

    /* Members */
    clsHeavyFootprint.def("isHeavy", &Class::isHeavy);
    clsHeavyFootprint.def(
            "insert", (void (Class::*)(lsst::afw::image::MaskedImage<ImagePixelT> &) const) & Class::insert);
    clsHeavyFootprint.def("insert",
                          (void (Class::*)(lsst::afw::image::Image<ImagePixelT> &) const) & Class::insert);
    clsHeavyFootprint.def("getImageArray",
                          (ndarray::Array<ImagePixelT, 1, 1> (Class::*)()) & Class::getImageArray);
    clsHeavyFootprint.def("getMaskArray",
                          (ndarray::Array<MaskPixelT, 1, 1> (Class::*)()) & Class::getMaskArray);
    clsHeavyFootprint.def("getVarianceArray",
                          (ndarray::Array<VariancePixelT, 1, 1> (Class::*)()) & Class::getVarianceArray);
    clsHeavyFootprint.def("getMaskBitsSet", &Class::getMaskBitsSet);
    clsHeavyFootprint.def("dot", &Class::dot);

    /* Module level */
    mod.def("makeHeavyFootprint",
            (Class(*)(Footprint const &,
                      lsst::afw::image::MaskedImage<ImagePixelT, MaskPixelT, VariancePixelT> const &,
                      HeavyFootprintCtrl const *))makeHeavyFootprint<ImagePixelT, MaskPixelT, VariancePixelT>,
            "foot"_a, "img"_a, "ctrl"_a = nullptr);

    // In Swig this seems to be suffixed with a type (i.e. mergeHeavyFootprintsF)
    // but there really doesn't seem any good reason why that is done, so removed it
    mod.def("mergeHeavyFootprints", mergeHeavyFootprints<ImagePixelT, MaskPixelT, VariancePixelT>);
}
}  // namespace

PYBIND11_MODULE(heavyFootprint, mod) {
    declareHeavyFootprint<int>(mod, "I");
    declareHeavyFootprint<std::uint16_t>(mod, "U");
    declareHeavyFootprint<float>(mod, "F");
    declareHeavyFootprint<double>(mod, "D");
}
}
}
}  // lsst::afw::detection