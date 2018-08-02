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
#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst {
namespace afw {
namespace geom {
namespace ellipses {

Ellipse::ParameterVector const Ellipse::getParameterVector() const {
    ParameterVector r;
    r.head<3>() = _core->getParameterVector();
    r.tail<2>() = _center.asEigen();
    return r;
}

void Ellipse::setParameterVector(Ellipse::ParameterVector const& vector) {
    _core->setParameterVector(vector.head<3>());
    _center = lsst::geom::Point2D(vector.tail<2>());
}

void Ellipse::readParameters(double const* iter) {
    _core->readParameters(iter);
    _center.setX(iter[3]);
    _center.setY(iter[4]);
}

void Ellipse::writeParameters(double* iter) const {
    _core->writeParameters(iter);
    iter[3] = _center.getX();
    iter[4] = _center.getY();
}

lsst::geom::Box2D Ellipse::computeBBox() const {
    lsst::geom::Extent2D dimensions = getCore().computeDimensions();
    return lsst::geom::Box2D(getCenter() - dimensions * 0.5, dimensions, false);
}

Ellipse& Ellipse::operator=(Ellipse const& other) {
    _center = other.getCenter();
    *_core = other.getCore();
    return *this;
}
// Delegate to copy-assignment for backwards compatibility
Ellipse& Ellipse::operator=(Ellipse&& other) { return *this = other; }

}  // namespace ellipses
}  // namespace geom
}  // namespace afw
}  // namespace lsst
