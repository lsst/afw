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

#ifndef LSST_AFW_GEOM_ELLIPSES_PixelRegion_h_INCLUDED
#define LSST_AFW_GEOM_ELLIPSES_PixelRegion_h_INCLUDED

#include "lsst/geom/Box.h"
#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

/**
 *  A lazy representation of the set of pixels whose centers are within an Ellipse.
 */
class PixelRegion {
public:
    class Iterator;

    //@{
    /// Iterate over the Spans covered by the Ellipse.
    Iterator begin() const;
    Iterator end() const;
    //@}

    /**
     * Return a pixel bounding box for the ellipse.
     *
     * The returned box is permitted to include pixels whose centers are not
     * within the ellipse.
     */
    lsst::geom::Box2I const& getBBox() const { return _bbox; }

    /**
     * Return a Span representing the extent of the ellipse for the given row.
     */
    Span const getSpanAt(int y) const;

    /**
     * Construct a new PixelRegion from the given Ellipse.
     */
    explicit PixelRegion(Ellipse const& ellipse);

    PixelRegion(PixelRegion const&) = default;
    PixelRegion(PixelRegion&&) = default;
    PixelRegion& operator=(PixelRegion const&) = default;
    PixelRegion& operator=(PixelRegion&&) = default;
    ~PixelRegion() = default;

private:
    lsst::geom::Point2D _center;
    double _detQ;
    double _invQxx;
    double _alpha;
    lsst::geom::Box2I _bbox;
};


/**
 * A generating iterator for the Spans covered by an Ellipse.
 *
 * This class provides most of the functionality of BidirectionalIterator, but
 * does not make a multi-pass guarantee, and is hence formally only an
 * InputIterator.
 */
class PixelRegion::Iterator {
public:

    using value_type = Span;
    using reference = Span const &;
    using pointer = Span const *;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    explicit Iterator(Span const& s = Span(0, 0, 0), PixelRegion const* region = NULL)
            : _s(s), _region(region) {}

    Iterator(Iterator const&) = default;
    Iterator(Iterator&&) = default;
    Iterator& operator=(Iterator const&) = default;
    Iterator& operator=(Iterator&&) = default;
    ~Iterator() = default;

    bool operator==(Iterator const & other) const {
        return _s == other._s;
    }

    bool operator!=(Iterator const & other) const {
        return !(*this == other);
    }

    reference operator*() const {
        return _s;
    }

    pointer operator->() const {
        return &_s;
    }

    Iterator & operator++() {
        _s = _region->getSpanAt(_s.getY() + 1);
        return *this;
    }

    Iterator & operator--() {
        _s = _region->getSpanAt(_s.getY() - 1);
        return *this;
    }

    Iterator operator++(int) {
        Iterator copy(*this);
        ++(this);
        return copy;
    }

    Iterator operator--(int) {
        Iterator copy(*this);
        --(this);
        return copy;
    }

private:
    Span _s;
    PixelRegion const* _region;
};

inline PixelRegion::Iterator PixelRegion::begin() const {
    return Iterator(getSpanAt(_bbox.getBeginY()), this);
}

inline PixelRegion::Iterator PixelRegion::end() const { return Iterator(getSpanAt(_bbox.getEndY()), this); }

}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst

#endif  // !LSST_AFW_GEOM_ELLIPSES_PixelRegion_h_INCLUDED
