#include <lsst/afw/math/ellipses/Ellipse.h>
#include <lsst/afw/math/ellipses/Quadrupole.h>
#include <lsst/afw/math/ellipses/Axes.h>
#include <lsst/afw/math/ellipses/Distortion.h>
#include <lsst/afw/math/ellipses/LogShear.h>

namespace ellipses = lsst::afw::math::ellipses;

ellipses::Distortion const & ellipses::DistortionEllipse::getCore() const {
    return static_cast<Distortion const &>(*_core); 
}

ellipses::Distortion & ellipses::DistortionEllipse::getCore() { 
    return static_cast<Distortion &>(*_core); 
}

void ellipses::DistortionEllipse::setComplex(std::complex<double> const & e) { 
    getCore().setComplex(e); 
}

std::complex<double> ellipses::DistortionEllipse::getComplex() const { 
    return getCore().getComplex(); 
}

void ellipses::DistortionEllipse::setE(double e) { getCore().setE(e); }
double ellipses::DistortionEllipse::getE() const { return getCore().getE(); }
   
ellipses::DistortionEllipse::DistortionEllipse(
        lsst::afw::math::Coordinate const & center
) : Ellipse(new Distortion(), center) {}

template <typename Derived>
ellipses::DistortionEllipse::DistortionEllipse(
        Eigen::MatrixBase<Derived> const & vector
) : Ellipse(vector.segment<2>(0)) {
    _core.reset(new Distortion(vector.segment<3>(2)));
}

ellipses::DistortionEllipse::DistortionEllipse(
        ellipses::Distortion const & core, 
        lsst::afw::math::Coordinate const & center
) : Ellipse(core,center) {}

ellipses::DistortionEllipse::DistortionEllipse(
        ellipses::Ellipse const & other
) : Ellipse(new Distortion(other.getCore()), other.getCenter()) {}

ellipses::DistortionEllipse::DistortionEllipse(
    ellipses::DistortionEllipse const & other
) : Ellipse(new Distortion(other.getCore()), other.getCenter()) {}


ellipses::Distortion::Ellipse * ellipses::Distortion::makeEllipse(
    lsst::afw::math::Coordinate const & center
) const {
    return new Ellipse(*this,center);
}

void ellipses::Distortion::assignTo(ellipses::Distortion & other) const {
    other._vector = this->_vector;
}

void ellipses::Distortion::assignTo(ellipses::Quadrupole & other) const {
    double e = getE();
    e *= e;
    double v = _vector[R] * _vector[R] / std::sqrt(1.0-e);
    other[Quadrupole::IXX] = v*(1.0+_vector[E1]);
    other[Quadrupole::IYY] = v*(1.0-_vector[E1]);
    other[Quadrupole::IXY] = v*_vector[E2];
}

void ellipses::Distortion::assignTo(ellipses::Axes & other) const {
    double e = getE();
    double q = std::pow((1.0-e)/(1.0+e),0.25);
    other[Axes::A] = _vector[R]/q;
    other[Axes::B] = _vector[R]*q;
    other[Axes::THETA] = 0.5*std::atan2(_vector[E2],_vector[E1]);
}

void ellipses::Distortion::assignTo(ellipses::LogShear & other) const {
    double e = getE();
    double gamma = 0.25*std::log((1.0+e)/(1.0-e));
    if (e < 1E-12) {
        other[LogShear::GAMMA1] = 0.0;
        other[LogShear::GAMMA2] = 0.0;
    } else {
        other[LogShear::GAMMA1] = _vector[E1]*gamma/e;
        other[LogShear::GAMMA2] = _vector[E2]*gamma/e;
    }
    other[LogShear::KAPPA] = std::log(_vector[R]);
}

Eigen::Matrix3d ellipses::Distortion::differentialAssignTo(
    ellipses::Distortion & other
) const {
    other._vector = this->_vector;
    return Eigen::Matrix3d::Identity();
}

Eigen::Matrix3d ellipses::Distortion::differentialAssignTo(
    ellipses::Quadrupole & other
) const {
    Eigen::Matrix3d m;
    double e = getE(); e *= e;
    double e1p = 1.0 + _vector[E1];
    double e1m = 1.0 - _vector[E1];
    double e2 = _vector[E2];
    double f = 1.0/(1.0-e);
    double g = _vector[R] * std::sqrt(f);
    m.setConstant(g);
    f *= _vector[R];
    m.col(0) *= f;
    m.col(1) *= f;
    m.col(2) *= 2.0;
    m(0,0) *= (e1p - e2*e2);
    m(0,1) *= e2*e1p;
    m(0,2) *= e1p;
    m(1,0) *= -(e1m - e2*e2);
    m(1,1) *= e2*e1m;
    m(1,2) *= e1m;
    m(2,0) *= _vector[E1]*e2;
    m(2,1) *= e1m*e1p;;
    m(2,2) *= e2;
    g *= _vector[R];
    other[Quadrupole::IXX] = g*e1p;
    other[Quadrupole::IYY] = g*e1m;
    other[Quadrupole::IXY] = g*e2;
    return m;
}

Eigen::Matrix3d ellipses::Distortion::differentialAssignTo(
    ellipses::Axes & other
) const {
    Eigen::Matrix3d m;
    double e = getE();
    double dp = std::pow(1.0 + e,0.25);
    double dm = std::pow(1.0 - e,0.25);
    other[Axes::A] = _vector[R]*dp/dm;
    other[Axes::B] = _vector[R]*dm/dp;
    other[Axes::THETA] = 0.5*std::atan2(_vector[E2],_vector[E1]);
    m(2,0) = m(2,1) = 1.0 / (2*e*e);
    m(2,0) *= -_vector[E2];
    m(2,1) *= _vector[E1];
    m(2,2) = 0.0;
    m(0,2) = dp/dm;
    m(1,2) = dm/dp;
    m.corner<2,2>(Eigen::TopLeft).setConstant(
        _vector[R] / (2 * e * std::pow(dp*dm,3))
    );
    dm *= dm;
    dp *= dp;
    m(0,0) *= _vector[E1] / dm;
    m(0,1) *= _vector[E2] / dm;
    m(1,0) *= -_vector[E1] / dp;
    m(1,1) *= -_vector[E2] / dp;
    return m;
}

Eigen::Matrix3d ellipses::Distortion::differentialAssignTo(
    ellipses::LogShear & other
) const {
    Eigen::Matrix3d m;
    double e = getE();
    double dp = 1.0 + e;
    double dm = 1.0 - e;
    double gamma = 0.25*std::log(dp/dm);
    double f1 = 0.5 / (dp*dm);
    double f2 = gamma / e;
    double d1 = _vector[E1] / e;
    double d2 = _vector[E2] / e;
    if (e < 1E-12) {
        other[LogShear::GAMMA1] = 0.0;
        other[LogShear::GAMMA2] = 0.0;
    } else {
        other[LogShear::GAMMA1] = _vector[E1]*f2;
        other[LogShear::GAMMA2] = _vector[E2]*f2;
    }
    other[LogShear::KAPPA] = std::log(_vector[R]);
    m(0,0) = d1*d1*f1 + d2*d2*f2;
    m(0,1) = m(1,0) = d1*d2*(f1 - f2);
    m(1,1) = d2*d2*f1 + d1*d1*f2;
    m(0,2) = m(2,0) = m(1,2) = m(2,1) = 0.0;
    m(2,2) = 1.0 / _vector[R];
    return m;
}
