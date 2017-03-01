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

#include "lsst/afw/table/io/python.h"  // for declarePersistableFacade

#include <iostream>
#include "lsst/afw/detection/Footprint.h"

namespace py = pybind11;
using namespace py::literals;

namespace lsst {
namespace afw {
namespace detection {

namespace {
    // Name chosen to match INSTANTIATE_MASK macro in src/detection/Footprints.cc
    template<typename PixelT, typename PyClass>
    void declareMask(PyClass & cls) {
        cls.def("insertIntoImage", (void (Footprint::*)(typename lsst::afw::image::Image<PixelT>&, std::uint64_t const, geom::Box2I const&) const) &Footprint::insertIntoImage<PixelT>,
                "idImage"_a, "id"_a, "region"_a=geom::Box2I());
        cls.def("insertIntoImage", (void (Footprint::*)(typename lsst::afw::image::Image<PixelT>&, std::uint64_t const, bool const, long const, typename std::set<std::uint64_t> *, geom::Box2I const&) const) &Footprint::insertIntoImage<PixelT>,
                "idImage"_a, "id"_a, "overwriteId"_a, "idMask"_a, "oldIds"_a, "region"_a=geom::Box2I());
        cls.def("overlapsMask", &Footprint::overlapsMask<PixelT>);
    }

    template<typename PixelT, typename PyClass>
    void declareNumericForClass(PyClass & cls) {
        cls.def("clipToNonzero", &Footprint::clipToNonzero<PixelT>);
    }

    template <typename PixelT>
    void declareNumeric(py::module & mod) {
        mod.def("setImageFromFootprint", (PixelT (*)(image::Image<PixelT> *, Footprint const&, PixelT const)) setImageFromFootprint<image::Image<PixelT>>);
        mod.def("setImageFromFootprintList", (PixelT (*)(image::Image<PixelT> *, std::vector<PTR(Footprint)> const&, PixelT const)) setImageFromFootprintList<image::Image<PixelT>>);
//        mod.def("copyWithinFootprint", copyWithinFootprint<image::Image<PixelT>>);
//        mod.def("copyWithinFootprint", copyWithinFootprint<image::MaskedImage<PixelT>>);
        mod.def("copyWithinFootprintImage", copyWithinFootprint<image::Image<PixelT>>);
        mod.def("copyWithinFootprintMaskedImage", copyWithinFootprint<image::MaskedImage<PixelT>>);
    }
}

PYBIND11_PLUGIN(_footprint) {
    py::module mod("_footprint", "Python wrapper for afw _footprint library");

    table::io::python::declarePersistableFacade<Footprint>(mod, "Footprint");

    py::class_<Footprint, std::shared_ptr<Footprint>, lsst::daf::base::Citizen, afw::table::io::PersistableFacade<Footprint>, afw::table::io::Persistable> clsFootprint(mod, "Footprint");

    clsFootprint.def(py::init<int, geom::Box2I const &>(),
            "nspan"_a=0, "region"_a=geom::Box2I());
    clsFootprint.def(py::init<afw::table::Schema const &, int, geom::Box2I const &>(),
            "peakSchema"_a, "nspan"_a=0, "region"_a=geom::Box2I());
    clsFootprint.def(py::init<geom::Box2I const &, geom::Box2I const &>(),
            "bbox"_a, "region"_a=geom::Box2I());
    clsFootprint.def(py::init<geom::Point2I const &, double const, geom::Box2I const &>(),
            "center"_a, "radius"_a, "region"_a=geom::Box2I());
    clsFootprint.def(py::init<geom::ellipses::Ellipse const &, geom::Box2I const &>(),
            "ellipse"_a, "region"_a=geom::Box2I());
    clsFootprint.def(py::init<typename Footprint::SpanList const &, geom::Box2I const &>(),
            "spans"_a, "region"_a);
    clsFootprint.def(py::init<Footprint const &>());

    clsFootprint.def("assign", &Footprint::operator=);

    // Default argument for bitmask doesn't work here, presumably because the Python `int`
    // generated doesn't cast to the image::MaskPixel type. But I am not 100% sure.
    // Therefore, this workaround until it can be investigated in more detail (DM-9401).
    clsFootprint.def("intersectMask", [](Footprint &self, image::Mask<image::MaskPixel> const &mask) {
        return self.intersectMask(mask);
    });
    clsFootprint.def("intersectMask",
                     [](Footprint &self, image::Mask<image::MaskPixel> const &mask,
                        image::MaskPixel bitmask) { return self.intersectMask(mask, bitmask); });
    clsFootprint.def("isHeavy", &Footprint::isHeavy);
    clsFootprint.def("getId", &Footprint::getId);
    clsFootprint.def("getSpans", (typename Footprint::SpanList& (Footprint::*)()) &Footprint::getSpans);
    clsFootprint.def("getPeaks", (PeakCatalog & (Footprint::*)()) &Footprint::getPeaks, py::return_value_policy::reference_internal);
    clsFootprint.def("addPeak", &Footprint::addPeak);
    clsFootprint.def("sortPeaks", &Footprint::sortPeaks,
            "key"_a=afw::table::Key<float>());
    clsFootprint.def("setPeakSchema", &Footprint::setPeakSchema);
    clsFootprint.def("getNpix", &Footprint::getNpix);
    clsFootprint.def("getArea", &Footprint::getArea);
    clsFootprint.def("getCentroid", &Footprint::getCentroid);
    clsFootprint.def("getShape", &Footprint::getShape);
    clsFootprint.def("addSpan", (const Span& (Footprint::*)(const int, const int, const int)) &Footprint::addSpan);
    clsFootprint.def("addSpan", (const Span& (Footprint::*)(Span const &)) &Footprint::addSpan);
    clsFootprint.def("addSpan", (const Span& (Footprint::*)(Span const &, int, int)) &Footprint::addSpan);
    clsFootprint.def("addSpanInSeries", &Footprint::addSpanInSeries);
    clsFootprint.def("shift", (void (Footprint::*)(int, int)) &Footprint::shift);
    clsFootprint.def("shift", (void (Footprint::*)(geom::ExtentI)) &Footprint::shift);
    clsFootprint.def("getBBox", &Footprint::getBBox);
    clsFootprint.def("getRegion", &Footprint::getRegion);
    clsFootprint.def("setRegion", &Footprint::setRegion);
    clsFootprint.def("clipTo", &Footprint::clipTo);
    clsFootprint.def("contains", &Footprint::contains);
    clsFootprint.def("normalize", &Footprint::normalize);
    clsFootprint.def("isNormalized", &Footprint::isNormalized);
    clsFootprint.def("transform", &Footprint::transform,
            "source"_a, "target"_a, "region"_a, "doClip"_a=true);
    clsFootprint.def("findEdgePixels", &Footprint::findEdgePixels);
    clsFootprint.def("include", &Footprint::include,
            "children"_a, "ignore"_a=false);
    clsFootprint.def("isPersistable", &Footprint::isPersistable);

    declareMask<int>(clsFootprint);
    declareMask<std::uint16_t>(clsFootprint);
    declareMask<std::uint64_t>(clsFootprint);

    declareNumericForClass<int>(clsFootprint);
    declareNumericForClass<float>(clsFootprint);
    declareNumericForClass<double>(clsFootprint);
    declareNumericForClass<std::uint16_t>(clsFootprint);
    declareNumericForClass<std::uint64_t>(clsFootprint);

    /* Member types and enums */

    /* Constructors */

    /* Operators */

    /* Members */

    /* Module level */
    declareNumeric<int>(mod);
    declareNumeric<float>(mod);
    declareNumeric<double>(mod);
    declareNumeric<std::uint16_t>(mod);
    declareNumeric<std::uint64_t>(mod);

    mod.def("nearestFootprint", nearestFootprint);
    mod.def("mergeFootprints", (PTR(Footprint) (*)(Footprint const&, Footprint const&)) mergeFootprints);
    mod.def("mergeFootprints", (PTR(Footprint) (*)(Footprint&, Footprint&)) mergeFootprints);
    mod.def("shrinkFootprint", shrinkFootprint);
    mod.def("growFootprint", (PTR(Footprint) (*)(Footprint const&, int, bool)) growFootprint,
            "foot"_a, "nGrow"_a, "isotropic"_a=true);
    mod.def("growFootprint", (PTR(Footprint) (*)(PTR(Footprint) const&, int, bool)) growFootprint,
            "foot"_a, "nGrow"_a, "isotropic"_a=true);
    mod.def("growFootprint", (PTR(Footprint) (*)(Footprint const&, int, bool, bool, bool, bool)) growFootprint);
    mod.def("footprintToBBoxList", footprintToBBoxList);
//
//template<typename ImageT>
//typename ImageT::Pixel setImageFromFootprint(ImageT *image,
//                                             Footprint const& footprint,
//                                             typename ImageT::Pixel const value);
//template<typename ImageT>
//typename ImageT::Pixel setImageFromFootprintList(ImageT *image,
//                                                 CONST_PTR(std::vector<PTR(Footprint)>) footprints,
//                                                 typename ImageT::Pixel  const value);
//template<typename ImageT>
//typename ImageT::Pixel setImageFromFootprintList(ImageT *image,
//                                                 std::vector<PTR(Footprint)> const& footprints,
//                                                 typename ImageT::Pixel  const value);
    mod.def("setMaskFromFootprint", (image::MaskPixel (*)(lsst::afw::image::Mask<image::MaskPixel> *, Footprint const&, image::MaskPixel const)) setMaskFromFootprint<image::MaskPixel>);
//template<typename MaskT>
//MaskT setMaskFromFootprint(lsst::afw::image::Mask<MaskT> *mask,
//                           Footprint const& footprint,
//                           MaskT const bitmask);
    mod.def("clearMaskFromFootprint", (image::MaskPixel (*)(lsst::afw::image::Mask<image::MaskPixel> *, Footprint const&, image::MaskPixel const)) clearMaskFromFootprint<image::MaskPixel>);
//template<typename MaskT>

//template <typename ImageOrMaskedImageT>
//void copyWithinFootprint(Footprint const& foot,
//                         PTR(ImageOrMaskedImageT) const input,
//                         PTR(ImageOrMaskedImageT) output);
    mod.def("setMaskFromFootprintList", (image::MaskPixel (*)(lsst::afw::image::Mask<image::MaskPixel> *, std::vector<PTR(Footprint)> const&, image::MaskPixel const)) setMaskFromFootprintList<image::MaskPixel>);
//                               MaskT const bitmask);
//template<typename MaskT>
//MaskT setMaskFromFootprintList(lsst::afw::image::Mask<MaskT> *mask,
//                               std::vector<PTR(Footprint)> const& footprints,
//                               MaskT const bitmask);
//template<typename MaskT>
//MaskT setMaskFromFootprintList(lsst::afw::image::Mask<MaskT> *mask,
//                               CONST_PTR(std::vector<PTR(Footprint)>) const& footprints,
//                               MaskT const bitmask);
//template<typename MaskT>
//PTR(Footprint) footprintAndMask(PTR(Footprint) const& foot,
//                                typename image::Mask<MaskT>::Ptr const& mask,
//                                MaskT const bitmask);
    return mod.ptr();
}

}}} // lsst::afw::detection