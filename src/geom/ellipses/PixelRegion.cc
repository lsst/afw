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

#include <cmath>
#include <limits>
#include "boost/optional.hpp"
#include "lsst/afw/geom/ellipses/PixelRegion.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

//
// Functor for computing the x boundary points of an ellipse given y.
//
// The equation for the boundary of the ellipse is:
//
//   F_{xx} x^2 + 2 F_{xy} x y + F_{yy} y^2 = 1
//
// where F is the matrix inverse of the quadrupole matrix Q and x and y are
// relative to the center.
//
// Given some fixed y, we want to solve this pretty vanilla quadratic equation
// for x.  Algebra is not shown, but the solution is:
//
//   x = p y ± \sqrt{r - t y^2}
//
// where p = -F_{xy}/F_{xx}  = Q_{xy}/Q_{yy}
//       r = 1/F_{xx}        = det(Q)/Q_{yy}   = Q_{xx} - Q_{xy}p
//       t = det(F)/F_{xx}^2 = det(Q)/Q_{yy}^2 = r/Q_{yy}
//
// This runs into divide-by-zero problems as Q_{yy} approaches zero.  That
// corresponds to the ellipse approaching a horizontal line segment.  The
// equation for that is just
//
//   x = ± \sqrt{r}
//
// with r = Q_{xx}.  That's equivalent to just setting p and t to zero.
//
class EllipseHorizontalLineIntersection {
public:

    explicit EllipseHorizontalLineIntersection(Ellipse const& ellipse) :
        _center(ellipse.getCenter()),
        _p(0.0), _t(0.0), _r(0.0)
    {
        Quadrupole converted(ellipse.getCore());
        converted.normalize();
        auto q = converted.getMatrix();
        if (q(1, 1) < q(0, 0)*std::numeric_limits<double>::epsilon()) {
            _r = q(0, 0);
        } else {
            _p = q(0, 1)/q(1, 1);
            _r = q(0, 0) - q(0, 1)*_p;
            _t = _r/q(1, 1);
        }
    }

    boost::optional<std::pair<double, double>> xAt(double y) const {
        double yc = y - _center.getY();
        double xc = _p*yc + _center.getX();
        double s = _r - _t*yc*yc;
        if (s < 0) {
            return boost::none;
        } else {
            double d = std::sqrt(s);
            return std::make_pair(xc - d, xc + d);
        }
    }

private:
    lsst::geom::Point2D _center;
    double _p;
    double _t;
    double _r;
};

PixelRegion::PixelRegion(Ellipse const& ellipse)
    : _bbox(ellipse.computeBBox(), lsst::geom::Box2I::EXPAND)
{
    // Initial temporary bounding box that may be larger than the final one.
    lsst::geom::Box2I envelope(ellipse.computeBBox(), lsst::geom::Box2I::EXPAND);

    if (envelope.isEmpty()) {
        // If the outer bbox is empty, we know there can't be any spans that
        // contain ellipse pixels.
        return;
    }

    // Helper class that does the hard work: compute the boundary points of the
    // ellipse in x that intersect a horizontal line at some given y.
    EllipseHorizontalLineIntersection intersection(ellipse);

    // Iterate over pixel rows in the bounding box, computing the intersection
    // of the ellipse with that y coordinate.
    int const yEnd = envelope.getEndY();
    for (int y = envelope.getBeginY(); y != yEnd; ++y) {
        auto x = intersection.xAt(y);
        if (x) {
            int xMin = std::ceil(x->first);
            int xMax = std::floor(x->second);
            if (xMax < xMin) continue;
            _spans.emplace_back(y, xMin, xMax);
            _bbox.include(lsst::geom::Point2I(xMin, y));
            _bbox.include(lsst::geom::Point2I(xMax, y));
        }
    }
}

Span const PixelRegion::getSpanAt(int y) const {
    if (_bbox.isEmpty()) {
        throw LSST_EXCEPT(
            pex::exceptions::OutOfRangeError,
            "PixelRegion is empty."
        );
    }
    if (y < _bbox.getMinY() || y > _bbox.getMaxY()) {
        throw LSST_EXCEPT(
            pex::exceptions::OutOfRangeError,
            (boost::format("No span at y=%s in pixel region with rows between y=%s and y=%s")
            % y % _bbox.getMinY() % _bbox.getMaxY()).str()
        );
    }
    return _spans[y - _bbox.getBeginY()];
}

}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst
