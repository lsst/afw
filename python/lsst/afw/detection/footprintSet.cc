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
void declareConstructors(PyClass &cls)
{
    cls.def(py::init<image::Image<PixelT> const &, Threshold const &, int const, bool const>(), "img"_a,
            "threshold"_a, "npixMin"_a = 1, "setPeaks"_a = true);
    cls.def(py::init<image::MaskedImage<PixelT, image::MaskPixel> const &, Threshold const &,
                     std::string const &, int const, bool const>(),
            "img"_a, "threshold"_a, "planeName"_a = "", "npixMin"_a = 1, "setPeaks"_a = true);
}

//    template <typename ImagePixelT, typename MaskPixelT>
//    void declareConstructors2(PyClass & cls) {
//    cls.def(py::init<image::Mask<PixelT> const&, Threshold const&, int const>(), "img"_a, "threshold"_a,
//            "npixMin"_a = 1);
//    }
}

PYBIND11_PLUGIN(_footprintSet)
{
    py::module mod("_footprintSet", "Python wrapper for afw _footprintSet library");

    py::class_<FootprintSet, lsst::daf::base::Citizen> clsFootprintSet(mod, "FootprintSet");

    /* Constructors */
    declareConstructors<std::uint16_t>(clsFootprintSet);
    declareConstructors<int>(clsFootprintSet);
    declareConstructors<float>(clsFootprintSet);
    declareConstructors<double>(clsFootprintSet);

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
    clsFootprintSet.def("getFootprints", [](FootprintSet & self) { return *(self.getFootprints()); });
    clsFootprintSet.def("makeSources", &FootprintSet::makeSources);
    clsFootprintSet.def("setRegion", &FootprintSet::setRegion);
    clsFootprintSet.def("getRegion", &FootprintSet::getRegion);
    clsFootprintSet.def("insertIntoImage", &FootprintSet::insertIntoImage);
    //
    //    template <typename MaskPixelT>
    //    void setMask(
    //        image::Mask<MaskPixelT> *mask, ///< Set bits in the mask
    //        std::string const& planeName   ///< Here's the name of the mask plane to fit
    //    )
    //    template <typename MaskPixelT>
    //    void setMask(
    //        PTR(image::Mask<MaskPixelT>) mask, ///< Set bits in the mask
    //        std::string const& planeName   ///< Here's the name of the mask plane to fit
    //    )
    //
    clsFootprintSet.def("merge", &FootprintSet::merge);
    //
    //    template <typename ImagePixelT, typename MaskPixelT>
    //    void makeHeavy(image::MaskedImage<ImagePixelT, MaskPixelT> const& mimg,
    //                   HeavyFootprintCtrl const* ctrl=NULL
    //                  );
    /* Module level */

    /* Member types and enums */

    return mod.ptr();
}
}}} // lsst::afw::detection
