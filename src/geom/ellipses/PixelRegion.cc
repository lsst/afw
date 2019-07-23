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
#include "lsst/afw/geom/ellipses/PixelRegion.h"
#include "lsst/afw/geom/ellipses/Quadrupole.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

PixelRegion::PixelRegion(Ellipse const& ellipse)
        : _bbox(ellipse.computeBBox(), lsst::geom::Box2I::EXPAND) {
    Quadrupole::Matrix q = Quadrupole(ellipse.getCore()).getMatrix();
    double const detQ = q(0, 0) * q(1, 1) - q(0, 1) * q(0, 1);
    double const invQxx = q(1, 1) / detQ;
    double alpha = q(0, 1) / detQ / invQxx;  // == -invQxy / invQxx
    if (std::isnan(alpha)) {
        alpha = 0.0;
    }
    lsst::geom::Point2D const center = ellipse.getCenter();
    int const yEnd = _bbox.getEndY();
    for (int y = _bbox.getBeginY(); y != yEnd; ++y) {
        double yt = y - center.getY();
        double d = invQxx - yt * yt / detQ;
        double x0 = center.getX() + yt * alpha;
        double x1 = x0;
        if (d > 0.0) {
            d = std::sqrt(d) / invQxx;
            x0 -= d;
            x1 += d;
        }  // Note that we return an empty span when d <= 0.0 or d is NaN.
        _spans.emplace_back(y, std::ceil(x0), std::floor(x1));
    }
}

Span const PixelRegion::getSpanAt(int y) const {
    int i = y - _bbox.getBeginY();
    if (i < 0 || std::size_t(i) >= _spans.size()) {
        return Span(y, 0, -1);
    }
    return _spans[i];
}

}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst
