// -*- lsst-c++ -*-
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Axes.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/LogShear.h"

namespace ellipses = lsst::afw::geom::ellipses;

ellipses::AxesEllipse::AxesEllipse(ParameterVector const & vector, bool doNormalize) :
    BaseEllipse(new Axes(vector.segment<3>(0)), PointD(vector.segment<2>(2))) 
{ 
    if (doNormalize) normalize(); 
}

bool ellipses::Axes::normalize() {
    if (_vector[A] < 0 || _vector[B] < 0) return false;
    if (_vector[A] < _vector[B]) {
        std::swap(_vector[A],_vector[B]);
        _vector[THETA] += M_PI_2;
    }
    if (_vector[THETA] > M_PI_2) {
        _vector[THETA] -= M_PI*std::ceil(_vector[THETA] / M_PI);
    } else if (_vector[THETA] < -M_PI_2) {
        _vector[THETA] += M_PI*std::ceil(-_vector[THETA] / M_PI);
    }
    return true;
}

lsst::afw::geom::LinearTransform ellipses::Axes::getGenerator() const {
    return LinearTransform::makeRotation(_vector[THETA]) * 
        LinearTransform::makeScaling(_vector[A], _vector[B]);
    
}

void ellipses::Axes::_assignTo(Axes & other) const {
    other._vector = this->_vector;
}

void ellipses::Axes::_assignTo(Quadrupole & other) const {
    double a = _vector[A];
    a *= a;
    double b = _vector[B];
    b *= b;
    double c = std::cos(_vector[THETA]);
    double s = std::sin(_vector[THETA]);
    other[Quadrupole::IXY] = (a - b)*c*s;
    c *= c;
    s *= s;
    other[Quadrupole::IXX] = c*a + s*b;
    other[Quadrupole::IYY] = s*a + c*b;
}

void ellipses::Axes::_assignTo(Distortion & other) const {
    Axes self(*this);
    self.normalize();
    double a = self[A];
    double b = self[B];
    other[Distortion::R] = std::sqrt(a*b);
    a *= a;
    b *= b;
    double e = (a-b)/(a+b);
    if (a+b == 0.0) e = 0.0;
    other[Distortion::E1] = e * std::cos(2*self[THETA]);
    other[Distortion::E2] = e * std::sin(2*self[THETA]);
}

void ellipses::Axes::_assignTo(LogShear & other) const {
    Axes self(*this);
    self.normalize();
    other[LogShear::KAPPA] = 0.5*std::log(self[A]*self[B]);
    double gamma = 0.5*(std::log(self[A]) - std::log(self[B]));
    other[LogShear::GAMMA1] = gamma*std::cos(2.0*self[THETA]);
    other[LogShear::GAMMA2] = gamma*std::sin(2.0*self[THETA]);
}

ellipses::BaseCore::Jacobian ellipses::Axes::_dAssignTo(Axes & other) const {
    other._vector = this->_vector;
    return BaseCore::Jacobian::Identity();
}

ellipses::BaseCore::Jacobian ellipses::Axes::_dAssignTo(Quadrupole & other) const {
    BaseCore::Jacobian m;
    double a = _vector[A];
    double b = _vector[B];
    m.col(0).setConstant(2*a);
    m.col(1).setConstant(2*b);
    a *= a;
    b *= b;
    m.col(2).setConstant(a-b);
    double c = std::cos(_vector[THETA]);
    double s = std::sin(_vector[THETA]);
    double cs = c*s;
    other[Quadrupole::IXY] = (a - b)*c*s;
    c *= c;
    s *= s;
    other[Quadrupole::IXX] = c*a + s*b;
    other[Quadrupole::IYY] = s*a + c*b;
    m(0,0) *= c;  m(0,1) *= s;   m(0,2) *= -2.0*cs;
    m(1,0) *= s;  m(1,1) *= c;   m(1,2) *= 2.0*cs;
    m(2,0) *= cs; m(2,1) *= -cs; m(2,2) *= (c - s);
    return m;
}

ellipses::BaseCore::Jacobian ellipses::Axes::_dAssignTo(Distortion & other) const {
    BaseCore::Jacobian m;
    Axes self(*this);
    self.normalize();
    double a = self[A];
    double b = self[B];
    double c = std::cos(2.0*self[THETA]);
    double s = std::sin(2.0*self[THETA]);
    other[Distortion::R] = std::sqrt(a*b);
    m(0,0) = m(1,0) = a*b*b;
    m(0,1) = m(1,1) = a*a*b;
    m(0,0) *= c;
    m(0,1) *= -c;
    m(1,0) *= s;
    m(1,1) *= -s;
    m(2,0) = 0.5 * std::sqrt(b/a);
    m(2,1) = 0.25 / m(2,0);
    a *= a;
    b *= b;
    double e = (a-b)/(a+b);
    other[Distortion::E1] = e * c;
    other[Distortion::E2] = e * s;
    double f = 2.0/(a+b);
    m.corner<2,2>(Eigen::TopLeft) *= f*f;
    m(0,2) = (b-a)*s*f;
    m(1,2) = (a-b)*c*f;
    m(2,2) = 0.0;
    return m;
}

ellipses::BaseCore::Jacobian ellipses::Axes::_dAssignTo(LogShear & other) const {
    Axes self(*this);
    self.normalize();
    double a = self[A];
    double b = self[B];
    double c = std::cos(2.0*self[THETA]);
    double s = std::sin(2.0*self[THETA]);
    double gamma = 0.5*std::log(a/b);        
    other[LogShear::KAPPA] = 0.5*std::log(a*b);
    other[LogShear::GAMMA1] = gamma*c;
    other[LogShear::GAMMA2] = gamma*s;
    BaseCore::Jacobian m;
    a *= 2.0;
    b *= 2.0;
    m(0,0) = c/a;
    m(0,1) = -c/b;
    m(0,2) = -2.0*gamma*s;
    m(1,0) = s/a;
    m(1,1) = -s/b;
    m(1,2) = 2.0*gamma*c;
    m(2,0) = 1.0/a;
    m(2,1) = 1.0/b;
    m(2,2) = 0.0;
    return m;
}
