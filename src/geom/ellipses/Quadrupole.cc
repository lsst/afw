// -*- lsst-c++ -*-
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Axes.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/LogShear.h"

namespace ellipses = lsst::afw::geom::ellipses;

ellipses::QuadrupoleEllipse::QuadrupoleEllipse(ParameterVector const & vector, bool doNormalize) :
    BaseEllipse(new Quadrupole(vector.segment<3>(0)), PointD(vector.segment<2>(2))) 
{ 
    if (doNormalize) normalize(); 
}

void ellipses::Quadrupole::_assignTo(Quadrupole & other) const {
    other._vector = this->_vector;
}

void ellipses::Quadrupole::_assignTo(Axes & other) const {
    double xx_p_yy = _vector[IXX] + _vector[IYY];
    double xx_m_yy = _vector[IXX] - _vector[IYY];
    double t = std::sqrt(xx_m_yy*xx_m_yy + 4*_vector[IXY]*_vector[IXY]);
    other[Axes::A] = std::sqrt(0.5*(xx_p_yy+t));
    other[Axes::B] = std::sqrt(0.5*(xx_p_yy-t));
    other[Axes::THETA] = 0.5*std::atan2(2.0*_vector[IXY],xx_m_yy);
}

void ellipses::Quadrupole::_assignTo(Distortion & other) const {
    double t = _vector[IXX] + _vector[IYY];
    if (t < 1E-12) {
        other[Distortion::E1] = other[Distortion::E2] = other[Distortion::R] = 0.0;
    } else {
        other[Distortion::E1] = (_vector[IXX]-_vector[IYY])/t;
        other[Distortion::E2] = 2*_vector[IXY]/t;
        other[Distortion::R] = std::pow(getDeterminant(),0.25);
    }
}

void ellipses::Quadrupole::_assignTo(LogShear & other) const {
    Distortion tmp;
    this->_assignTo(tmp);
    static_cast<BaseCore &>(tmp)._assignTo(other);
}

ellipses::BaseCore::Jacobian ellipses::Quadrupole::_dAssignTo(Quadrupole & other) const {
    other._vector = this->_vector;
    return BaseCore::Jacobian::Identity();
}

ellipses::BaseCore::Jacobian ellipses::Quadrupole::_dAssignTo(Axes & other) const {
    Distortion tmp;
    BaseCore::Jacobian a = this->_dAssignTo(tmp);
    BaseCore::Jacobian b = static_cast<BaseCore&>(tmp)._dAssignTo(other);
    return b*a;
}

ellipses::BaseCore::Jacobian ellipses::Quadrupole::_dAssignTo(Distortion & other) const {
    _assignTo(other);
    BaseCore::Jacobian m;
    double d = 1.0 / (_vector[IXX] + _vector[IYY]);
    m(0,2) = 0.0;
    m(1,2) = 2.0 * d;
    m.corner<2,2>(Eigen::TopLeft).setConstant(2.0*d*d);
    m.row(2).setConstant(std::pow(getDeterminant(),-0.75));
    m(0,0) *= _vector[IYY];
    m(0,1) *= -_vector[IXX];
    m(1,0) *= -_vector[IXY];
    m(1,1) *= -_vector[IXY];
    m(2,0) *= _vector[IYY]/4;
    m(2,1) *= _vector[IXX]/4;
    m(2,2) *= -_vector[IXY]/2;
    return m;
}

ellipses::BaseCore::Jacobian ellipses::Quadrupole::_dAssignTo(LogShear & other) const {
    Distortion tmp;
    BaseCore::Jacobian a = this->_dAssignTo(tmp);
    BaseCore::Jacobian b = static_cast<BaseCore&>(tmp)._dAssignTo(other);
    return b*a;
}
