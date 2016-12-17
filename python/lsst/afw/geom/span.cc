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
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/SpanPixelIterator.h"

namespace py = pybind11;

using namespace lsst::afw::geom;

// A thin wrapper around SpanPixelIterator.
// Unfortunately we cannot use py::make_iterator here, as we normally
// should, because for some reason the return values then all refer
// to the last element.
class SpanIterator {
    public:
        SpanIterator(const Span &s) : _it{s.begin()}, _end{s.end()} {};
        Point2I next() {
            if (_it == _end) {
                throw py::stop_iteration();
            }
            return *_it++;
        };
    private:
        Span::Iterator _it;
        Span::Iterator _end;
};

PYBIND11_PLUGIN(_span) {
    py::module mod("_span", "Python wrapper for afw _span library");

    py::class_<SpanIterator> clsSpanIterator(mod, "SpanIterator");

    /* Operators */
    clsSpanIterator.def("__iter__", [](SpanIterator &it) -> SpanIterator& { return it; });
    clsSpanIterator.def("__next__", &SpanIterator::next);

    py::class_<Span, std::shared_ptr<Span>> clsSpan(mod, "Span");

    /* Constructors */
    clsSpan.def(py::init<int, int, int>());
    clsSpan.def(py::init<>());

    /* Operators */
    clsSpan.def(py::self == py::self);
    clsSpan.def(py::self != py::self);
    clsSpan.def(py::self < py::self);
    clsSpan.def("__len__", &Span::getWidth);
    clsSpan.def("__str__", &Span::toString);
    clsSpan.def("__iter__", [](const Span &s) { return SpanIterator(s); },
        py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */);

    /* Members */
    clsSpan.def("getX0", (int (Span::*)() const) &Span::getX0);
    clsSpan.def("getX1", (int (Span::*)() const) &Span::getX1);
    clsSpan.def("getY", (int (Span::*)() const) &Span::getY);
    clsSpan.def("getWidth", &Span::getWidth);
    clsSpan.def("getMinX", &Span::getMinX);
    clsSpan.def("getMaxX", &Span::getMaxX);
    clsSpan.def("getBeginX", &Span::getBeginX);
    clsSpan.def("getEndX", &Span::getEndX);
    clsSpan.def("getMin", &Span::getMin);
    clsSpan.def("getMax", &Span::getMax);
    clsSpan.def("contains", (bool (Span::*)(int) const) &Span::contains);
    clsSpan.def("contains", (bool (Span::*)(int, int) const) &Span::contains);
    clsSpan.def("contains", (bool (Span::*)(Point2I const &) const) &Span::contains);
    clsSpan.def("isEmpty", &Span::isEmpty);
    clsSpan.def("toString", &Span::toString);
    clsSpan.def("shift", &Span::shift);

    return mod.ptr();
}

