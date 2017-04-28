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
#include <pybind11/stl.h>

#include "lsst/afw/detection/FootprintSet.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

namespace {
template <typename PixelT, typename PyClass>
void declareMakeHeavy(PyClass &cls) {
    //    cls.def("makeHeavy", [](FootprintSet & self, image::MaskedImage<PixelT, image::MaskPixel> const&
    //    mimg) {
    //            return self.makeHeavy(mimg);
    //            });
    //    cls.def("makeHeavy", [](FootprintSet & self, image::MaskedImage<PixelT, image::MaskPixel> const&
    //    mimg, HeavyFootprintCtrl const* ctrl) {
    //            return self.makeHeavy(mimg, ctrl);
    //            });
    cls.def("makeHeavy", (void (FootprintSet::*)(image::MaskedImage<PixelT, image::MaskPixel> const &,
                                                 HeavyFootprintCtrl const *)) &
                                 FootprintSet::makeHeavy<PixelT, image::MaskPixel>,
            "mimg"_a, "ctrl"_a = nullptr);
}

template <typename PixelT, typename PyClass>
void declareSetMask(PyClass &cls) {
    cls.def("setMask", (void (FootprintSet::*)(image::Mask<PixelT> *, std::string const &)) &
                               FootprintSet::setMask<PixelT>,
            "mask"_a, "planeName"_a);
}

template <typename PixelT, typename PyClass>
void declareTemplatedMembers(PyClass &cls) {
    /* Constructors */
    cls.def(py::init<image::Image<PixelT> const &, Threshold const &, int const, bool const>(), "img"_a,
            "threshold"_a, "npixMin"_a = 1, "setPeaks"_a = true);
    cls.def(py::init<image::MaskedImage<PixelT, image::MaskPixel> const &, Threshold const &,
                     std::string const &, int const, bool const>(),
            "img"_a, "threshold"_a, "planeName"_a = "", "npixMin"_a = 1, "setPeaks"_a = true);

    /* Members */
    declareMakeHeavy<int>(cls);
    declareMakeHeavy<float>(cls);
    declareMakeHeavy<double>(cls);
    declareMakeHeavy<std::uint16_t>(cls);
    //    declareMakeHeavy<std::uint64_t>(cls);
    declareSetMask<image::MaskPixel>(cls);
}
}  // namespace

PYBIND11_PLUGIN(_footprintSet) {
    py::module mod("_footprintSet", "Python wrapper for afw _footprintSet library");

    py::class_<FootprintSet, std::shared_ptr<FootprintSet>, lsst::daf::base::Citizen> clsFootprintSet(
            mod, "FootprintSet");

    declareTemplatedMembers<std::uint16_t>(clsFootprintSet);
    declareTemplatedMembers<int>(clsFootprintSet);
    declareTemplatedMembers<float>(clsFootprintSet);
    declareTemplatedMembers<double>(clsFootprintSet);

    clsFootprintSet.def(py::init<image::Mask<image::MaskPixel> const &, Threshold const &, int const>(),
                        "img"_a, "threshold"_a, "npixMin"_a = 1);

    /* Members */
    clsFootprintSet.def(py::init<geom::Box2I>(), "region"_a);
    clsFootprintSet.def(py::init<FootprintSet const &>(), "set"_a);
    clsFootprintSet.def(py::init<FootprintSet const &, int, FootprintControl const &>(), "set"_a, "rGrow"_a,
                        "ctrl"_a);
    clsFootprintSet.def(py::init<FootprintSet const &, int, bool>(), "set"_a, "rGrow"_a, "isotropic"_a);
    clsFootprintSet.def(py::init<FootprintSet const &, FootprintSet const &, bool>(), "footprints1"_a,
                        "footprints2"_a, "includePeaks"_a);

    clsFootprintSet.def("swap", &FootprintSet::swap);
    //    clsFootprintSet.def("swapFootprintList", &FootprintSet::swapFootprintList);
    clsFootprintSet.def("setFootprints", &FootprintSet::setFootprints);
    // getFootprints returns shared_ptr<FootprintList>, but stl caster can't handle this
    clsFootprintSet.def("getFootprints", [](FootprintSet &self) { return *(self.getFootprints()); });
    clsFootprintSet.def("makeSources", &FootprintSet::makeSources);
    clsFootprintSet.def("setRegion", &FootprintSet::setRegion);
    clsFootprintSet.def("getRegion", &FootprintSet::getRegion);
    clsFootprintSet.def("insertIntoImage", &FootprintSet::insertIntoImage);
    clsFootprintSet.def("setMask", (void (FootprintSet::*)(image::Mask<lsst::afw::image::MaskPixel> *,
                                                           std::string const &)) &
                                           FootprintSet::setMask<lsst::afw::image::MaskPixel>);
    clsFootprintSet.def("setMask",
                        (void (FootprintSet::*)(std::shared_ptr<image::Mask<lsst::afw::image::MaskPixel>>,
                                                std::string const &)) &
                                FootprintSet::setMask<lsst::afw::image::MaskPixel>);
    clsFootprintSet.def("merge", &FootprintSet::merge, "rhs"_a, "tGrow"_a = 0, "rGrow"_a = 0,
                        "isotropic"_a = true);

    /* Module level */

    /* Member types and enums */

    return mod.ptr();
}
}
}
}  // lsst::afw::detection
