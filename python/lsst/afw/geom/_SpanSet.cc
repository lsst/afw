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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "numpy/arrayobject.h"
#include "ndarray/pybind11.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"

#include "lsst/afw/geom/SpanSet.h"


namespace py = pybind11;
using namespace pybind11::literals;

namespace lsst { namespace afw { namespace geom {

namespace {

using PySpanSet = py::class_<SpanSet,std::shared_ptr<SpanSet>>;

template <typename Pixel, typename PyClass>
void declareFlattenMethod(PyClass & cls) {
    cls.def("flatten", (ndarray::Array<Pixel, 1, 1> (SpanSet::*)(ndarray::Array<Pixel, 2, 0> const &,
                                                                 Point2I const &) const)
                                                                 &SpanSet::flatten<Pixel, 2, 0>,
                                                                 "input"_a,
                                                                 "xy0"_a = Point2I());
    cls.def("flatten", (ndarray::Array<Pixel, 2, 2> (SpanSet::*)(ndarray::Array<Pixel, 3, 0> const &,
                                                                 Point2I const &) const)
                                                                 &SpanSet::flatten<Pixel, 3, 0>,
                                                                 "input"_a,
                                                                 "xy0"_a = Point2I());
    cls.def("flatten", (void (SpanSet::*)(ndarray::Array<Pixel, 1, 0> const &,
                                          ndarray::Array<Pixel, 2, 0> const &,
                                          Point2I const &) const) &SpanSet::flatten<Pixel, Pixel, 2, 0, 0>,
                                          "output"_a,
                                          "input"_a,
                                          "xy0"_a = Point2I());
    cls.def("flatten", (void (SpanSet::*)(ndarray::Array<Pixel, 2, 0> const &,
                                          ndarray::Array<Pixel, 3, 0> const &,
                                          Point2I const &) const) &SpanSet::flatten<Pixel, Pixel, 3, 0, 0>,
                                          "output"_a,
                                          "input"_a,
                                          "xy0"_a = Point2I());
}

template <typename Pixel>
void declareUnflattenMethod(PySpanSet & cls) {
    cls.def("unflatten",
            (ndarray::Array<Pixel, 2, 2> (SpanSet::*)(ndarray::Array<Pixel, 1, 0> const & input) const)
            &SpanSet::unflatten<Pixel, 1, 0>);
    cls.def("unflatten",
            (ndarray::Array<Pixel, 3, 3> (SpanSet::*)(ndarray::Array<Pixel, 2, 0> const & input) const)
            &SpanSet::unflatten<Pixel, 2, 0>);
    cls.def("unflatten", (void (SpanSet::*)(ndarray::Array<Pixel, 2, 0> const & ,
                                            ndarray::Array<Pixel, 1, 0> const & ,
                                            Point2I const &) const) &SpanSet::unflatten<Pixel, Pixel, 1, 0, 0>,
                                            "output"_a,
                                            "input"_a,
                                            "xy0"_a = Point2I());
    cls.def("unflatten", (void (SpanSet::*)(ndarray::Array<Pixel, 3, 0> const & ,
                                            ndarray::Array<Pixel, 2, 0> const & ,
                                            Point2I const &) const) &SpanSet::unflatten<Pixel, Pixel, 2, 0, 0>,
                                            "output"_a,
                                            "input"_a,
                                            "xy0"_a = Point2I());
}

template <typename Pixel>
void declareSetMaskMethod(PySpanSet & cls) {
    cls.def("setMask", (void (SpanSet::*)(lsst::afw::image::Mask<Pixel> &, Pixel) const)
                       &SpanSet::setMask<Pixel>);
}

template <typename Pixel>
void declareClearMaskMethod(PySpanSet & cls) {
    cls.def("clearMask",
            (void (SpanSet::*)(lsst::afw::image::Mask<Pixel> &, Pixel) const) &SpanSet::clearMask<Pixel>);
}

template <typename Pixel>
void declareIntersectMethod(PySpanSet & cls) {
    cls.def("intersect", (std::shared_ptr<SpanSet> (SpanSet::*)(lsst::afw::image::Mask<Pixel> const &,
                                                                Pixel const &) const)
                                                                &SpanSet::intersect<Pixel>);
}

template <typename Pixel>
void declareIntersectNotMethod(PySpanSet & cls) {
    cls.def("intersectNot", (std::shared_ptr<SpanSet> (SpanSet::*)(lsst::afw::image::Mask<Pixel> const &,
                                                                   Pixel const &) const)
                                                                   &SpanSet::intersectNot<Pixel>);
}

template <typename Pixel>
void declareUnionMethod(PySpanSet & cls) {
    cls.def("union", (std::shared_ptr<SpanSet> (SpanSet::*)(lsst::afw::image::Mask<Pixel> const &,
                                                            Pixel const &) const)
                                                            &SpanSet::union_<Pixel>);
}

template <typename Pixel>
void declareMaskToSpanSetFunction(py::module & mod) {
    mod.def("maskToSpanSet", [](lsst::afw::image::Mask<Pixel> mask) { return maskToSpanSet(mask); });
    mod.def("maskToSpanSet", [](lsst::afw::image::Mask<Pixel> mask, Pixel const &bitmask) {
        auto functor = [&bitmask](Pixel const &pixval) { return (pixval & bitmask) == bitmask; };
        return maskToSpanSet(mask, functor);
    });
}

template <typename Pixel, typename PyClass>
void declareMaskMethods(PyClass & cls) {
    declareSetMaskMethod<Pixel>(cls);
    declareClearMaskMethod<Pixel>(cls);
    declareIntersectMethod<Pixel>(cls);
    declareIntersectNotMethod<Pixel>(cls);
    declareUnionMethod<Pixel>(cls);
}

} // end anonymous namespace

PYBIND11_PLUGIN(_SpanSet) {
    using MaskPixel = lsst::afw::image::MaskPixel;
    py::module mod("_SpanSet");

    py::module::import("lsst.afw.geom._Span");

    if(_import_array() < 0) {
        PyErr_SetString(PyExc_ImportError, "numpy.core.multiarray failed to import");
        return nullptr;
    }

    py::enum_<Stencil>(mod, "Stencil")
        .value("CIRCLE", Stencil::CIRCLE)
        .value("BOX", Stencil::BOX)
        .value("MANHATTAN", Stencil::MANHATTAN);

    PySpanSet cls(mod, "SpanSet");

    /* SpanSet Constructors */
    cls.def(py::init<>());
    cls.def(py::init<Box2I>(), "box"_a);
    cls.def(py::init<std::vector<Span>, bool>(), "spans"_a, "normalize"_a = true);

    /* SpanSet Methods */
    cls.def("getArea", &SpanSet::getArea);
    cls.def("getBBox", &SpanSet::getBBox);
    cls.def("isContiguous", &SpanSet::isContiguous);
    cls.def("shiftedBy", (std::shared_ptr<SpanSet>(SpanSet::*)(int, int) const) & SpanSet::shiftedBy);
    cls.def("shiftedBy", (std::shared_ptr<SpanSet>(SpanSet::*)(Extent2I const &) const) & SpanSet::shiftedBy);
    cls.def("clippedTo", &SpanSet::clippedTo);
    cls.def("transformedBy",
            (std::shared_ptr<SpanSet>(SpanSet::*)(LinearTransform const &) const) & SpanSet::transformedBy);
    cls.def("transformedBy",
            (std::shared_ptr<SpanSet>(SpanSet::*)(AffineTransform const &) const) & SpanSet::transformedBy);
    cls.def("transformedBy",
            (std::shared_ptr<SpanSet>(SpanSet::*)(XYTransform const &) const) & SpanSet::transformedBy);
    cls.def("overlaps", &SpanSet::overlaps);
    cls.def("contains", (bool (SpanSet::*)(SpanSet const &) const) & SpanSet::contains);
    cls.def("contains", (bool (SpanSet::*)(Point2I const &) const) & SpanSet::contains);
    cls.def("computeCentroid", &SpanSet::computeCentroid);
    cls.def("computeShape", &SpanSet::computeShape);
    cls.def("dilate", (std::shared_ptr<SpanSet>(SpanSet::*)(int, Stencil) const) & SpanSet::dilate, "r"_a,
            "s"_a = Stencil::CIRCLE);
    cls.def("dilate", (std::shared_ptr<SpanSet>(SpanSet::*)(SpanSet const &) const) & SpanSet::dilate);
    cls.def("erode", (std::shared_ptr<SpanSet>(SpanSet::*)(int, Stencil) const) & SpanSet::erode, "r"_a,
            "s"_a = Stencil::CIRCLE);
    cls.def("erode", (std::shared_ptr<SpanSet>(SpanSet::*)(SpanSet const &) const) & SpanSet::erode);
    cls.def("intersect", (std::shared_ptr<SpanSet>(SpanSet::*)(SpanSet const &) const) & SpanSet::intersect);
    cls.def("intersectNot",
            (std::shared_ptr<SpanSet>(SpanSet::*)(SpanSet const &) const) & SpanSet::intersectNot);
    cls.def("union", (std::shared_ptr<SpanSet>(SpanSet::*)(SpanSet const &) const) & SpanSet::union_);
    cls.def_static("spanSetFromShape",
                   (std::shared_ptr<SpanSet>(*)(int, Stencil)) & SpanSet::spanSetFromShape);
    cls.def("split", &SpanSet::split);
    cls.def("findEdgePixels", &SpanSet::findEdgePixels);

    /* SpanSet Operators */
    cls.def("__eq__", [](SpanSet const &self, SpanSet const &other) -> bool { return self == other; },
            py::is_operator());
    cls.def("__ne__", [](SpanSet const &self, SpanSet const &other) -> bool { return self != other; },
            py::is_operator());
    cls.def("__iter__", [](SpanSet &self) { return py::make_iterator(self.begin(), self.end()); },
            py::keep_alive<0, 1>());
    cls.def("__len__", [](SpanSet const &self) -> decltype(self.size()) { return self.size(); });
    cls.def("__contains__", [](SpanSet &self, SpanSet const &other) -> bool { return self.contains(other); });
    cls.def("__contains__", [](SpanSet &self, Point2I &other) -> bool { return self.contains(other); });
    cls.def("__repr__", [](SpanSet const &self) -> std::string {
        std::ostringstream os;
        lsst::afw::image::Mask<MaskPixel> tempMask(self.getBBox());
        self.setMask(tempMask, static_cast<MaskPixel>(1));
        auto array = tempMask.getArray();
        auto dims = array.getShape();
        for (std::size_t i = 0; i < dims[0]; ++i) {
            os << "[";
            for (std::size_t j = 0; j < dims[1]; ++j) {
                os << array[i][j];
                if (j != dims[1] - 1) {
                    os << ", ";
                }
            }
            os << "]" << std::endl;
        }
        return os.str();
    });
    cls.def("__str__", [](SpanSet const &self) -> std::string {
        std::ostringstream os;
        for (auto const &span : self) {
            os << span.getY() << ": " << span.getMinX() << ".." << span.getMaxX() << std::endl;
        }
        return os.str();
    });
    // Instantiate all the templates
    declareFlattenMethod<uint16_t>(cls);
    declareFlattenMethod<uint64_t>(cls);
    declareFlattenMethod<int>(cls);
    declareFlattenMethod<long>(cls);
    declareFlattenMethod<float>(cls);
    declareFlattenMethod<double>(cls);

    declareUnflattenMethod<uint16_t>(cls);
    declareUnflattenMethod<uint64_t>(cls);
    declareUnflattenMethod<int>(cls);
    declareUnflattenMethod<long>(cls);
    declareUnflattenMethod<float>(cls);
    declareUnflattenMethod<double>(cls);

    declareMaskMethods<MaskPixel>(cls);

    /* Free Functions */
    declareMaskToSpanSetFunction<MaskPixel>(mod);

    return mod.ptr();
}

}}} // end lsst::afw::geom
