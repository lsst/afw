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
 
#include "lsst/afw/geom/ellipses/BaseEllipse.h"
#include "lsst/afw/geom/ellipses/Axes.h"
#include "lsst/afw/geom/ellipses/Transformer.h"

namespace ellipses = lsst::afw::geom::ellipses;

/**
 *  \brief Set the parameters of this ellipse from another.
 *
 *  This does not change the parametrization of the ellipse.
 */
ellipses::BaseEllipse & ellipses::BaseEllipse::operator=(BaseEllipse const & other) { 
    _center = other.getCenter();
    *_core = other.getCore();
    return *this;
}

/**
 * Return the ellipse parameters as a vector.
 */
ellipses::BaseEllipse::ParameterVector const ellipses::BaseEllipse::getVector() const {
    ParameterVector r;
    r << _center[X], _center[Y], (*_core)[0], (*_core)[1], (*_core)[2];
    return r;
}
/**
 * Set the ellipse parameters from a vector.
 */
void ellipses::BaseEllipse::setVector(BaseEllipse::ParameterVector const & vector) {
    _center = PointD(vector.segment<2>(0));
    _core->setVector(vector.segment<3>(2));
}

/**
 * Return the AffineTransform that transforms the unit circle at the origin 
 * into this.
 */
lsst::afw::geom::AffineTransform ellipses::BaseEllipse::getGenerator() const {
    AffineTransform r(_core->getGenerator());
    r[AffineTransform::X] = _center.getX();
    r[AffineTransform::Y] = _center.getY();
    return r;
}


/**
 * Return the bounding box of the ellipse.
 */
ellipses::BaseEllipse::Envelope ellipses::BaseEllipse::computeEnvelope() const {
    ExtentD size(getCore().computeDimensions());
    return Envelope(_center - size * 0.5, size);
}

lsst::afw::geom::ExtentD ellipses::BaseCore::computeDimensions() const {
    Axes axes(*this);
    double c = std::cos(axes[Axes::THETA]);
    double s = std::sin(axes[Axes::THETA]);
    c *= c;
    s *= s;
    double b2 = axes[Axes::B] * axes[Axes::B];
    double a2 = axes[Axes::A] * axes[Axes::A];
    double as2 = a2*s;
    double bc2 = b2*c;
    ExtentD dimensions = ExtentD::make(std::sqrt(b2*s+a2*c),std::sqrt(as2+bc2));
    dimensions *= 2;
    return dimensions;
}

void ellipses::BaseCore::grow(double buffer) {
    Axes axes(*this);
    axes.grow(buffer);
    *this = axes;
}

lsst::afw::geom::LinearTransform ellipses::BaseCore::getGenerator() const {
    Axes tmp(*this);
    return tmp.getGenerator();
}
