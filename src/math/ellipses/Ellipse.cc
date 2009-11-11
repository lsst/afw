#include <lsst/afw/math/ellipses/Ellipse.h>
#include <lsst/afw/math/ellipses/Axes.h>

namespace ellipses = lsst::afw::math::ellipses;

ellipses::Ellipse::TransformDerivative::TransformDerivative(
    ellipses::Ellipse const & ellipse, 
    lsst::afw::math::AffineTransform const & transform
) : _center(ellipse.getCenter()), _core_d(ellipse.getCore(),transform) {}

Eigen::Matrix5d ellipses::Ellipse::TransformDerivative::dInput() const {
    Eigen::Matrix5d r = Eigen::Matrix5d::Zero();
    r.block<2,2>(0,0) = _core_d.transform().matrix().linear();
    r.block<3,3>(2,2) = _core_d.dInput();
    return r;
}

Eigen::Matrix<double,5,6> ellipses::Ellipse::TransformDerivative::dTransform() const {
    Eigen::Matrix<double,5,6> r = Eigen::Matrix<double,5,6>::Zero();
    r.block<2,6>(0,0) = _core_d.transform().d(_center);
    r.block<3,6>(2,0) = _core_d.dTransform();
    return r;
}

void ellipses::Ellipse::transform(
    lsst::afw::math::AffineTransform const & transform
) {
    _center = transform(_center);
    _core->transform(transform);
}

lsst::afw::math::AffineTransform ellipses::Ellipse::getGenerator() const {
    AffineTransform r = _core->getGenerator();
    r.matrix().translation() << _center.getX(), _center.getY();
    return r;
}

#if 0
ellipses::Ellipse::Rectangle::ConstPtr Ellipse::computeEnvelope() const {
    Axes axes(*_core);
    double c = std::cos(axes[Axes::THETA]);
    double s = std::sin(axes[Axes::THETA]);
    c *= c;
    s *= s;
    double b2 = axes[Axes::B] * axes[Axes::B];
    double a2 = axes[Axes::A] * axes[Axes::A];
    double as2 = a2*s;
    double bc2 = b2*c;
    double xm = std::sqrt(b2*s+a2*c);
    double ym = std::sqrt(as2+bc2);
    return Envelope::ConstPtr(new Envelope(_center,xm*2,ym*2));
}
#endif

ellipses::Parametric::Parametric(
    ellipses::Ellipse const& ellipse
) : _center(ellipse.getCenter()) {
    Axes core(ellipse.getCore());
    double c = std::cos(core[Axes::THETA]);
    double s = std::sin(core[Axes::THETA]);
    _u = lsst::afw::image::PointD(core[Axes::A]*c,core[Axes::A]*s);
    _v = lsst::afw::image::PointD(-core[Axes::B]*s,core[Axes::B]*c);
}
