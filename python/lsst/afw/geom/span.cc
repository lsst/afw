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

#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/SpanPixelIterator.h"

namespace py = pybind11;

namespace lsst {
namespace afw {
namespace geom {
namespace {

using PySpan = py::class_<Span, std::shared_ptr<Span>>;

// A thin wrapper around SpanPixelIterator.
// Unfortunately we cannot use py::make_iterator here, as we normally
// should, because for some reason the return values then all refer
// to the last element.
class SpanIterator {
public:
    SpanIterator(const Span &s) : _it{s.begin()}, _end{s.end()} {};
    lsst::geom::Point2I next() {
        if (_it == _end) {
            throw py::stop_iteration();
        }
        return *_it++;
    };

private:
    Span::Iterator _it;
    Span::Iterator _end;
};

static void declareSpanIterator(py::module &mod) {
    py::class_<SpanIterator> cls(mod, "SpanIterator");
    cls.def("__iter__", [](SpanIterator &it) -> SpanIterator & { return it; });
    cls.def("__next__", &SpanIterator::next);
}

PYBIND11_PLUGIN(span) {
    py::module mod("span");

    py::module::import("lsst.geom");

    declareSpanIterator(mod);

    PySpan cls(mod, "Span");
    cls.def(py::init<int, int, int>());
    cls.def(py::init<>());
    cls.def("__eq__", &Span::operator==, py::is_operator());
    cls.def("__ne__", &Span::operator!=, py::is_operator());
    cls.def("__lt__", &Span::operator<, py::is_operator());
    cls.def("__len__", &Span::getWidth);
    cls.def("__str__", &Span::toString);
    // unlike most iterators, SpanPixelIterator doesn't actually refer
    // back to its container (the Span), and there's no need to keep the
    // Span alive for the lifetime of the iterator.
    cls.def("__iter__", [](const Span &s) { return SpanIterator(s); });
    cls.def("getX0", (int (Span::*)() const) & Span::getX0);
    cls.def("getX1", (int (Span::*)() const) & Span::getX1);
    cls.def("getY", (int (Span::*)() const) & Span::getY);
    cls.def("getWidth", &Span::getWidth);
    cls.def("getMinX", &Span::getMinX);
    cls.def("getMaxX", &Span::getMaxX);
    cls.def("getBeginX", &Span::getBeginX);
    cls.def("getEndX", &Span::getEndX);
    cls.def("getMin", &Span::getMin);
    cls.def("getMax", &Span::getMax);
    cls.def("contains", (bool (Span::*)(int) const) & Span::contains);
    cls.def("contains", (bool (Span::*)(int, int) const) & Span::contains);
    cls.def("contains", (bool (Span::*)(lsst::geom::Point2I const &) const) & Span::contains);
    cls.def("isEmpty", &Span::isEmpty);
    cls.def("toString", &Span::toString);
    cls.def("shift", &Span::shift);

    return mod.ptr();
}
}
}
}
}  // namespace lsst::afw::geom::<anonymous>
