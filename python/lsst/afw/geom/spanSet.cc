
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "ndarray/converter.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"

#include "lsst/afw/geom/SpanSet.h"


namespace py = pybind11;

namespace lsst { namespace afw { namespace geom {

namespace {
    template <typename pixel, typename PyClass>
    void declareFlattenMethod(PyClass & cls) {
        cls.def("flatten", (ndarray::Array<pixel, 1, 1> (SpanSet::*)(ndarray::Array<pixel, 2, 2> &,
                                                                     Point2I const &) const)
                                                                     &SpanSet::flatten<pixel, 2>,
                                                                     py::arg("input"),
                                                                     py::arg("xy0") = Point2I());
        cls.def("flatten", (void (SpanSet::*)(ndarray::Array<pixel, 1, 1> const &,
                                              ndarray::Array<pixel, 2, 2> const &,
                                              Point2I const &) const) &SpanSet::flatten<pixel, 1, 2>,
                                              py::arg("output"),
                                              py::arg("input"),
                                              py::arg("xy0") = Point2I());
    }

    template <typename pixel, typename PyClass>
    void declareUnflattenMethod(PyClass & cls) {
        cls.def("unflatten",
                (ndarray::Array<pixel, 2, 2> (SpanSet::*)(ndarray::Array<pixel, 1, 1> & input) const)
                &SpanSet::unflatten<pixel, 1>);
        cls.def("unflatten", (void (SpanSet::*)(ndarray::Array<pixel, 2, 2> & ,
                                                ndarray::Array<pixel, 1, 1> & ,
                                                Point2I const &) const) &SpanSet::unflatten<pixel, 2, 1>,
                                                py::arg("output"),
                                                py::arg("input"),
                                                py::arg("xy0") = Point2I());
    }

    template <typename pixel, typename PyClass>
    void declareSetMaskMethod(PyClass & cls) {
        cls.def("setMask", (void (SpanSet::*)(lsst::afw::image::Mask<pixel> &, pixel) const)
                           &SpanSet::setMask<pixel>);
    }

    template <typename pixel, typename PyClass>
    void declareClearMaskMethod(PyClass & cls) {
        cls.def("clearMask",
                (void (SpanSet::*)(lsst::afw::image::Mask<pixel> &, pixel) const) &SpanSet::clearMask<pixel>);
    }

    template <typename pixel, typename PyClass>
    void declareIntersectMethod(PyClass & cls) {
        cls.def("intersect", (std::shared_ptr<SpanSet> (SpanSet::*)(lsst::afw::image::Mask<pixel> const &,
                                                                    pixel const &) const)
                                                                    &SpanSet::intersect<pixel>);
    }

    template <typename pixel, typename PyClass>
    void declareIntersectNotMethod(PyClass & cls) {
        cls.def("intersectNot", (std::shared_ptr<SpanSet> (SpanSet::*)(lsst::afw::image::Mask<pixel> const &,
                                                                       pixel const &) const)
                                                                       &SpanSet::intersectNot<pixel>);
    }

    template <typename pixel, typename PyClass>
    void declareUnionMethod(PyClass & cls) {
        cls.def("union", (std::shared_ptr<SpanSet> (SpanSet::*)(lsst::afw::image::Mask<pixel> const &,
                                                                pixel const &) const)
                                                                &SpanSet::union_<pixel>);
    }

    template <typename pixel>
    void declareMaskToSpanSetFunction(py::module & mod) {
        mod.def("maskToSpanSet",
                []
                (lsst::afw::image::Mask<pixel> mask)
                {
                    return maskToSpanSet(mask);
                });
        mod.def("maskToSpanSet",
                []
                (lsst::afw::image::Mask<pixel> mask, pixel const & bitmask)
                {
                    auto functor = [&bitmask](pixel const & pixval){ return (pixval & bitmask) == bitmask; };
                    return maskToSpanSet(mask, functor);
                });
    }

    template <typename pixel, typename PyClass>
    void declareMaskMethods(PyClass & cls) {
        declareSetMaskMethod<pixel>(cls);
        declareClearMaskMethod<pixel>(cls);
        declareIntersectMethod<pixel>(cls);
        declareIntersectNotMethod<pixel>(cls);
        declareUnionMethod<pixel>(cls);
    }

} // end anonymous namespace

PYBIND11_PLUGIN(_spanSet) {
    using MaskPixel = lsst::afw::image::MaskPixel;
    py::module mod("_spanSet", "Python wrapper for afw _spanSet library");

    if(_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::enum_<Stencil>(mod, "Stencil")
        .value("CIRCLE", Stencil::CIRCLE)
        .value("BOX", Stencil::BOX)
        .value("MANHATTAN", Stencil::MANHATTAN);

    py::class_<SpanSet, std::shared_ptr<SpanSet>> clsSpanSet(mod, "SpanSet");

    /* SpanSet Constructors */
    clsSpanSet.def(py::init<>());
    clsSpanSet.def(py::init<Box2I>());
    clsSpanSet.def(py::init<std::vector<Span>, bool>(), py::arg("spans"), py::arg("normalize") = true);

    /* SpanSet Methods */
    clsSpanSet.def("getArea", &SpanSet::getArea);
    clsSpanSet.def("getBBox", &SpanSet::getBBox);
    clsSpanSet.def("isContiguous", &SpanSet::isContiguous);
    clsSpanSet.def("shiftedBy", (std::shared_ptr<SpanSet> (SpanSet::*)(int, int) const) &SpanSet::shiftedBy);
    clsSpanSet.def("shiftedBy",
                   (std::shared_ptr<SpanSet> (SpanSet::*)(Extent2I const &) const) &SpanSet::shiftedBy);
    clsSpanSet.def("clippedTo", &SpanSet::clippedTo);
    clsSpanSet.def("transformedBy",
                   (std::shared_ptr<SpanSet> (SpanSet::*)(LinearTransform const &) const)
                   &SpanSet::transformedBy);
    clsSpanSet.def("transformedBy",
                   (std::shared_ptr<SpanSet> (SpanSet::*)(AffineTransform const &) const)
                   &SpanSet::transformedBy);
    clsSpanSet.def("transformedBy",
                   (std::shared_ptr<SpanSet> (SpanSet::*)(XYTransform const &) const)
                   &SpanSet::transformedBy);
    clsSpanSet.def("overlaps", &SpanSet::overlaps);
    clsSpanSet.def("contains", (bool (SpanSet::*)(SpanSet const &) const) &SpanSet::contains);
    clsSpanSet.def("contains", (bool (SpanSet::*)(Point2I const &) const) &SpanSet::contains);
    clsSpanSet.def("computeCentroid", &SpanSet::computeCentroid);
    clsSpanSet.def("computeShape", &SpanSet::computeShape);
    clsSpanSet.def("dilate", (std::shared_ptr<SpanSet> (SpanSet::*)(int, Stencil) const) &SpanSet::dilate,
                   py::arg("r"), py::arg("s") = Stencil::CIRCLE);
    clsSpanSet.def("dilate", (std::shared_ptr<SpanSet> (SpanSet::*)(SpanSet const &) const) &SpanSet::dilate);
    clsSpanSet.def("erode", (std::shared_ptr<SpanSet> (SpanSet::*)(int, Stencil) const) &SpanSet::erode,
                   py::arg("r"), py::arg("s") = Stencil::CIRCLE);
    clsSpanSet.def("erode", (std::shared_ptr<SpanSet> (SpanSet::*)(SpanSet const &) const) &SpanSet::erode);
    clsSpanSet.def("intersect",
                   (std::shared_ptr<SpanSet> (SpanSet::*)(SpanSet const &) const) &SpanSet::intersect);
    clsSpanSet.def("intersectNot",
                    (std::shared_ptr<SpanSet> (SpanSet::*)(SpanSet const &) const) &SpanSet::intersectNot);
    clsSpanSet.def("union", (std::shared_ptr<SpanSet> (SpanSet::*)(SpanSet const &) const) &SpanSet::union_);
    clsSpanSet.def_static("spanSetFromShape",
                          (std::shared_ptr<SpanSet> (*)(int, Stencil)) &SpanSet::spanSetFromShape);
    clsSpanSet.def("split", &SpanSet::split);
    clsSpanSet.def("findEdgePixels", &SpanSet::findEdgePixels);

    /* SpanSet Operators */
    clsSpanSet.def("__eq__",
                   [](SpanSet const & self, SpanSet const & other) -> bool {return self == other;},
                   py::is_operator());
    clsSpanSet.def("__ne__",
                   [](SpanSet const & self, SpanSet const & other) -> bool {return self != other;},
                   py::is_operator());
    clsSpanSet.def("__iter__",
                   [](SpanSet & self){return py::make_iterator(self.begin(), self.end());},
                   py::keep_alive<0, 1>());
    clsSpanSet.def("__len__",
                   [](SpanSet const & self) -> decltype(self.size()) { return self.size(); });
    clsSpanSet.def("__contains__",
                   [](SpanSet & self, SpanSet const & other)->bool {return self.contains(other); });
    clsSpanSet.def("__contains__",
                   [](SpanSet & self, Point2I & other)->bool {return self.contains(other); });
    // Instantiate all the templates
    declareFlattenMethod<uint16_t>(clsSpanSet);
    declareFlattenMethod<uint64_t>(clsSpanSet);
    declareFlattenMethod<int>(clsSpanSet);
    declareFlattenMethod<long>(clsSpanSet);
    declareFlattenMethod<float>(clsSpanSet);
    declareFlattenMethod<double>(clsSpanSet);

    declareUnflattenMethod<uint16_t>(clsSpanSet);
    declareUnflattenMethod<uint64_t>(clsSpanSet);
    declareUnflattenMethod<int>(clsSpanSet);
    declareUnflattenMethod<long>(clsSpanSet);
    declareUnflattenMethod<float>(clsSpanSet);
    declareUnflattenMethod<double>(clsSpanSet);

    declareMaskMethods<MaskPixel>(clsSpanSet);

    /* Free Functions */
    declareMaskToSpanSetFunction<MaskPixel>(mod);

    return mod.ptr();
}

}}} // end lsst::afw::geom
