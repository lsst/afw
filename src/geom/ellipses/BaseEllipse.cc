// -*- lsst-c++ -*-
#include "lsst/afw/geom/ellipses/BaseEllipse.h"
#include "lsst/afw/geom/ellipses/Axes.h"

namespace ellipses = lsst::afw::geom::ellipses;

ellipses::BaseEllipse & ellipses::BaseEllipse::operator=(BaseEllipse const & other) { 
    _center = other.getCenter();
    *_core = other.getCore();
    return *this;
}

ellipses::BaseEllipse::ParameterVector const ellipses::BaseEllipse::getVector() const {
    ParameterVector r;
    r << _center[X], _center[Y], (*_core)[0], (*_core)[1], (*_core)[2];
    return r;
}

void ellipses::BaseEllipse::setVector(BaseEllipse::ParameterVector const & vector) {
    _center = PointD(vector.segment<2>(0));
    _core->setVector(vector.segment<3>(2));
}

lsst::afw::geom::AffineTransform ellipses::BaseEllipse::getGenerator() const {
    AffineTransform r(_core->getGenerator());
    r.matrix().translation() = _center.asVector();
    return r;
}

ellipses::BaseEllipse::Envelope ellipses::BaseEllipse::computeEnvelope() const {
    return Envelope(_center,getCore().computeDimensions());
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
    ExtentD dimensions = ExtentD::makeXY(std::sqrt(b2*s+a2*c),std::sqrt(as2+bc2));
    dimensions *= 2;
    return dimensions;
}

void ellipses::BaseCore::grow(double buffer) {
    Axes axes(*this);
    axes.grow(buffer);
    *this = axes;
}

lsst::afw::geom::AffineTransform ellipses::BaseCore::getGenerator() const {
    Axes tmp(*this);
    return tmp.getGenerator();
}
