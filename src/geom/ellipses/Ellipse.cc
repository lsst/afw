// -*- lsst-c++ -*-

/* 
 * LSST Data Management System
 * See the COPYRIGHT and LICENSE files in the top-level directory of this
 * package for notices and licensing terms.
 */
#include "lsst/afw/geom/ellipses/Ellipse.h"

namespace lsst { namespace afw { namespace geom { namespace ellipses {

Ellipse::ParameterVector const Ellipse::getParameterVector() const {
    ParameterVector r;
    r.head<3>() = _core->getParameterVector();
    r.tail<2>() = _center.asEigen();
    return r;
}

void Ellipse::setParameterVector(Ellipse::ParameterVector const & vector) {
    _core->setParameterVector(vector.head<3>());
    _center = Point2D(vector.tail<2>());
}

void Ellipse::readParameters(double const * iter) {
    _core->readParameters(iter);
    _center.setX(iter[3]);
    _center.setY(iter[4]);
}

void Ellipse::writeParameters(double * iter) const {
    _core->writeParameters(iter);
    iter[3] = _center.getX();
    iter[4] = _center.getY();
}

Box2D Ellipse::computeBBox() const {
    Extent2D dimensions = getCore().computeDimensions();
    return Box2D(getCenter() - dimensions * 0.5, dimensions);
}

Ellipse & Ellipse::operator=(Ellipse const & other) { 
    _center = other.getCenter();
    *_core = other.getCore();
    return *this;
}

}}}} // namespace lsst::afw::geom::ellipses
