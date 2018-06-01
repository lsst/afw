
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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <cstdint>

#include "ndarray/pybind11.h"

#include "lsst/pex/exceptions.h"
#include "lsst/afw/table/io/python.h"  // for addPersistableMethods
#include "lsst/afw/detection/Footprint.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst {
namespace afw {
namespace detection {

namespace {

template <typename MaskT>
void declareMaskFromFootprintList(py::module &mod) {
    auto maskSetter = [](lsst::afw::image::Mask<MaskT> *mask,
                         std::vector<std::shared_ptr<lsst::afw::detection::Footprint>> const &footprints,
                         MaskT const bitmask, bool doClip) {
        for (auto const &foot : footprints) {
            try {
                if (doClip) {
                    auto tmpSpan = foot->getSpans()->clippedTo(mask->getBBox());
                    tmpSpan->setMask(*mask, bitmask);
                } else {
                    foot->getSpans()->setMask(*mask, bitmask);
                }
            } catch (lsst::pex::exceptions::OutOfRangeError e) {
                throw LSST_EXCEPT(lsst::pex::exceptions::OutOfRangeError,
                                  "Bounds of a Footprint fall outside mask set doClip to force");
            }
        }
    };
    mod.def("setMaskFromFootprintList", std::move(maskSetter), "mask"_a, "footprints"_a, "bitmask"_a,
            "doClip"_a = true);
}

}  // end anonymous namespace

PYBIND11_PLUGIN(_footprint) {
    py::module mod("_footprint", "Python wrapper for afw Footprint library");

    /* Footprint Constructors */
    py::class_<Footprint, std::shared_ptr<Footprint>, daf::base::Citizen> clsFootprint(mod, "Footprint");
    clsFootprint.def(py::init<std::shared_ptr<geom::SpanSet>, lsst::geom::Box2I const &>(), "inputSpans"_a,
                     "region"_a = lsst::geom::Box2I());

    clsFootprint.def(
            py::init<std::shared_ptr<geom::SpanSet>, afw::table::Schema const &, lsst::geom::Box2I const &>(),
            "inputSpans"_a, "peakSchema"_a, "region"_a = lsst::geom::Box2I());
    clsFootprint.def(py::init<Footprint const &>());
    clsFootprint.def(py::init<>());

    table::io::python::addPersistableMethods<Footprint>(clsFootprint);

    /* Footprint Methods */
    clsFootprint.def("getSpans", &Footprint::getSpans);
    clsFootprint.def("setSpans", &Footprint::setSpans);
    clsFootprint.def("getPeaks", (PeakCatalog & (Footprint::*)()) & Footprint::getPeaks,
                     py::return_value_policy::reference_internal);
    clsFootprint.def("addPeak", &Footprint::addPeak);
    clsFootprint.def("sortPeaks", &Footprint::sortPeaks, "key"_a = afw::table::Key<float>());
    clsFootprint.def("setPeakSchema", &Footprint::setPeakSchema);
    clsFootprint.def("getArea", &Footprint::getArea);
    clsFootprint.def("getCentroid", &Footprint::getCentroid);
    clsFootprint.def("getShape", &Footprint::getShape);
    clsFootprint.def("shift", (void (Footprint::*)(int, int)) & Footprint::shift);
    clsFootprint.def("shift", (void (Footprint::*)(lsst::geom::ExtentI const &)) & Footprint::shift);
    clsFootprint.def("getBBox", &Footprint::getBBox);
    clsFootprint.def("getRegion", &Footprint::getRegion);
    clsFootprint.def("setRegion", &Footprint::setRegion);
    clsFootprint.def("clipTo", &Footprint::clipTo);
    clsFootprint.def("contains", &Footprint::contains);
    clsFootprint.def("transform", (std::shared_ptr<Footprint> (Footprint::*)(
                                          std::shared_ptr<geom::SkyWcs>, std::shared_ptr<geom::SkyWcs>,
                                          lsst::geom::Box2I const &, bool) const) &
                                          Footprint::transform,
                     "source"_a, "target"_a, "region"_a, "doClip"_a = true);
    clsFootprint.def("transform", (std::shared_ptr<Footprint> (Footprint::*)(
                                          lsst::geom::LinearTransform const &, lsst::geom::Box2I const &, bool) const) &
                                          Footprint::transform);
    clsFootprint.def("transform", (std::shared_ptr<Footprint> (Footprint::*)(
                                          lsst::geom::AffineTransform const &, lsst::geom::Box2I const &, bool) const) &
                                          Footprint::transform);
    clsFootprint.def("transform",
                     (std::shared_ptr<Footprint>(Footprint::*)(geom::TransformPoint2ToPoint2 const &,
                                                               lsst::geom::Box2I const &, bool) const) &
                             Footprint::transform);
    clsFootprint.def("dilate", (void (Footprint::*)(int, geom::Stencil)) & Footprint::dilate, "r"_a,
                     "stencil"_a = geom::Stencil::CIRCLE);
    clsFootprint.def("dilate", (void (Footprint::*)(geom::SpanSet const &)) & Footprint::dilate);
    clsFootprint.def("erode", (void (Footprint::*)(int, geom::Stencil)) & Footprint::erode, "r"_a,
                     "stencil"_a = geom::Stencil::CIRCLE);
    clsFootprint.def("erode", (void (Footprint::*)(geom::SpanSet const &)) & Footprint::erode);
    clsFootprint.def("removeOrphanPeaks", &Footprint::removeOrphanPeaks);
    clsFootprint.def("isContiguous", &Footprint::isContiguous);
    clsFootprint.def("isHeavy", &Footprint::isHeavy);
    clsFootprint.def("assign", (Footprint & (Footprint::*)(Footprint const &)) & Footprint::operator=);
    clsFootprint.def("split", [](Footprint const &self) -> py::list {
        // This is a work around for pybind not properly
        // handling converting a vector of unique pointers
        // to python lists of shared pointers
        py::list l;
        for (auto &ptr : self.split()) {
            l.append(py::cast(std::shared_ptr<Footprint>(std::move(ptr))));
        }
        return l;
    });

    /* Define python level properties */
    clsFootprint.def_property("spans", &Footprint::getSpans, &Footprint::setSpans);
    clsFootprint.def_property_readonly("peaks", (PeakCatalog & (Footprint::*)()) & Footprint::getPeaks,
                                       py::return_value_policy::reference);

    /* Python Operators functions */
    clsFootprint.def("__contains__", [](Footprint const &self, lsst::geom::Point2I const &point) -> bool {
        return self.contains(point);
    });
    clsFootprint.def("__eq__",
                     [](Footprint const &self, Footprint const &other) -> bool { return self == other; },
                     py::is_operator());

    declareMaskFromFootprintList<lsst::afw::image::MaskPixel>(mod);

    mod.def("mergeFootprints", &mergeFootprints);
    mod.def("footprintToBBoxList", &footprintToBBoxList);

    return mod.ptr();
}
}
}
}  // close lsst::afw::detection
