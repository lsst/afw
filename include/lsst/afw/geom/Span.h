// -*- lsst-c++ -*-
/*
 * LSST Data Management System
 * Copyright 2008, 2009, 2010 LSST Corporation.
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
 * see <http://www.lsstcorp.org/LegalNotices/>.
 */

#ifndef LSST_AFW_GEOM_Span_h_INCLUDED
#define LSST_AFW_GEOM_Span_h_INCLUDED

#include <string>
#include <iostream>

#include "lsst/base.h"
#include "lsst/geom.h"
#include "lsst/afw/geom/SpanPixelIterator.h"

namespace lsst {
namespace afw {

namespace detection {
class Footprint;
}  // namespace detection

namespace geom {

/**
 * A range of pixels within one row of an Image
 */
class Span final {
public:
    /// An iterator over points in the Span.
    typedef SpanPixelIterator Iterator;

    Span(int y,   ///< Row that Span's in
         int x0,  ///< Starting column (inclusive)
         int x1)  ///< Ending column (inclusive)
            : _y(y), _x0(x0), _x1(x1) {}

    /// Construct an empty Span with zero width at the origin.
    Span() noexcept : _y(0), _x0(0), _x1(-1) {}

    Span(Span const&) noexcept = default;
    Span(Span&&) noexcept = default;
    Span& operator=(Span const&) noexcept = default;
    Span& operator=(Span&&) noexcept = default;
    ~Span() noexcept = default;

    /// Return an iterator to the first pixel in the Span.
    Iterator begin() const noexcept { return Iterator(lsst::geom::Point2I(_x0, _y)); }

    /// Return an iterator to one past the last pixel in the Span.
    Iterator end() const noexcept { return Iterator(lsst::geom::Point2I(_x1 + 1, _y)); }

    int getX0() const noexcept { return _x0; }               ///< Return the starting x-value
    int& getX0() noexcept { return _x0; }                    ///< Return the starting x-value
    int getX1() const noexcept { return _x1; }               ///< Return the ending x-value
    int& getX1() noexcept { return _x1; }                    ///< Return the ending x-value
    int getY() const noexcept { return _y; }                 ///< Return the y-value
    int& getY() noexcept { return _y; }                      ///< Return the y-value
    int getWidth() const noexcept { return _x1 - _x0 + 1; }  ///< Return the number of pixels
    int getMinX() const noexcept { return _x0; }             ///< Minimum x-value.
    int getMaxX() const noexcept { return _x1; }             ///< Maximum x-value.
    int getBeginX() const noexcept { return _x0; }           ///< Begin (inclusive) x-value.
    int getEndX() const noexcept { return _x1 + 1; }         ///< End (exclusive) x-value.
    lsst::geom::Point2I const getMin() const noexcept {
        return lsst::geom::Point2I(_x0, _y);
    }  ///< Point corresponding to minimum x.
    lsst::geom::Point2I const getMax() const noexcept {
        return lsst::geom::Point2I(_x1, _y);
    }  ///< Point corresponding to maximum x.

    bool contains(int x) const noexcept { return (x >= _x0) && (x <= _x1); }
    bool contains(int x, int y) const noexcept { return (x >= _x0) && (x <= _x1) && (y == _y); }
    bool contains(lsst::geom::Point2I const& point) const noexcept {
        return contains(point.getX(), point.getY());
    }

    /// Return true if the span contains no pixels.
    bool isEmpty() const noexcept { return _x1 < _x0; }

    /// Return a string-representation of a Span
    std::string toString() const;

    void shift(int dx, int dy) noexcept {
        _x0 += dx;
        _x1 += dx;
        _y += dy;
    }

    /// Stream output; delegates to toString().
    friend std::ostream& operator<<(std::ostream& os, Span const& span) { return os << span.toString(); }

    bool operator==(Span const& other) const noexcept {
        return other.getY() == getY() && other.getMinX() == getMinX() && other.getMaxX() == getMaxX();
    }
    bool operator!=(Span const& other) const noexcept { return !(*this == other); }

    /* Required to make Span "LessThanComparable" so they can be used
     * in sorting, binary search, etc.
     * http://www.sgi.com/tech/stl/LessThanComparable.html
     */
    bool operator<(const Span& b) const noexcept;

    friend class detection::Footprint;

private:
    int _y;   ///< Row that Span's in
    int _x0;  ///< Starting column (inclusive)
    int _x1;  ///< Ending column (inclusive)
};
}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // LSST_AFW_GEOM_Span_h_INCLUDED
