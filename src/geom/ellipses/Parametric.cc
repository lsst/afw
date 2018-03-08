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
#include "lsst/afw/geom/ellipses/Parametric.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

Parametric::Parametric(Ellipse const& ellipse) : _center(ellipse.getCenter()) {
    double a, b, theta;
    ellipse.getCore()._assignToAxes(a, b, theta);
    double c = std::cos(theta);
    double s = std::sin(theta);
    _u = Extent2D(a * c, a * s);
    _v = Extent2D(-b * s, b * c);
}
}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst
