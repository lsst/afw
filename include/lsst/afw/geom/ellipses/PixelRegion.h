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

#include "boost/iterator/iterator_facade.hpp"

#include "lsst/afw/geom/Span.h"
#include "lsst/afw/geom/Box.h"
#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

class PixelRegion {
public:
    class Iterator;

    Iterator begin() const;
    Iterator end() const;

    Box2I const& getBBox() const { return _bbox; }

    Span const getSpanAt(int y) const;

    explicit PixelRegion(Ellipse const& ellipse);

private:
    Point2D _center;
    double _detQ;
    double _invQxx;
    double _alpha;
    Box2I _bbox;
};

class PixelRegion::Iterator : public boost::iterator_facade<PixelRegion::Iterator, Span const,
                                                            boost::random_access_traversal_tag> {
public:
    explicit Iterator(Span const& s = Span(0, 0, 0), PixelRegion const* region = NULL)
            : _s(s), _region(region) {}

private:
    friend class boost::iterator_core_access;

    Span const& dereference() const { return _s; }

    void increment() { _s = _region->getSpanAt(_s.getY() + 1); }

    void decrement() { _s = _region->getSpanAt(_s.getY() - 1); }

    void advance(int n) { _s = _region->getSpanAt(_s.getY() + n); }

    bool equal(Iterator const& other) const { return _s == other._s; }

    int distance_to(Iterator const& other) const { return other._s.getY() - _s.getY(); }

    Span _s;
    PixelRegion const* _region;
};

inline PixelRegion::Iterator PixelRegion::begin() const {
    return Iterator(getSpanAt(_bbox.getBeginY()), this);
}
inline PixelRegion::Iterator PixelRegion::end() const { return Iterator(getSpanAt(_bbox.getEndY()), this); }
}
}
}
}  // namespace lsst::afw::geom::ellipses

#endif  // !LSST_AFW_GEOM_ELLIPSES_PixelRegion_h_INCLUDED
