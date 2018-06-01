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
        : _center(ellipse.getCenter()), _bbox(ellipse.computeBBox(), lsst::geom::Box2I::EXPAND) {
    Quadrupole::Matrix q = Quadrupole(ellipse.getCore()).getMatrix();
    _detQ = q(0, 0) * q(1, 1) - q(0, 1) * q(0, 1);
    _invQxx = q(1, 1) / _detQ;
    _alpha = q(0, 1) / _detQ / _invQxx;  // == -invQxy / invQxx
    if (std::isnan(_alpha)) {
        _alpha = 0.0;
    }
}

Span const PixelRegion::getSpanAt(int y) const {
    double yt = y - _center.getY();
    double d = _invQxx - yt * yt / _detQ;
    double x0 = _center.getX() + yt * _alpha;
    double x1 = x0;
    if (d > 0.0) {
        d = std::sqrt(d) / _invQxx;
        x0 -= d;
        x1 += d;
    }  // Note that we return an empty span when d <= 0.0 or d is NaN.
    return Span(y, std::ceil(x0), std::floor(x1));
}
}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst
