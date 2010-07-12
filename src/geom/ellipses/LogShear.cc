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
 
#include "lsst/afw/geom/ellipses/Quadrupole.h"
#include "lsst/afw/geom/ellipses/Axes.h"
#include "lsst/afw/geom/ellipses/Distortion.h"
#include "lsst/afw/geom/ellipses/LogShear.h"

namespace ellipses = lsst::afw::geom::ellipses;

ellipses::LogShearEllipse::LogShearEllipse(ParameterVector const & vector, bool doNormalize) :
    BaseEllipse(new LogShear(vector.segment<3>(0)), PointD(vector.segment<2>(2))) 
{ 
    if (doNormalize) normalize(); 
}

void ellipses::LogShear::_assignTo(LogShear & other) const {
    other._vector = this->_vector;
}

void ellipses::LogShear::_assignTo(Quadrupole & other) const {
    double exp_2kappa = std::exp(2.0*_vector[KAPPA]);
    double gamma = getGamma();
    double cosh_2gamma = std::cosh(2.0*gamma);
    double sinh_2gamma = std::sinh(2.0*gamma);
    if (gamma < 1E-12) 
        sinh_2gamma = 0;
    else
        sinh_2gamma /= gamma;
    other[Quadrupole::IXX] = exp_2kappa*(cosh_2gamma + _vector[GAMMA1]*sinh_2gamma);
    other[Quadrupole::IYY] = exp_2kappa*(cosh_2gamma - _vector[GAMMA1]*sinh_2gamma);
    other[Quadrupole::IXY] = exp_2kappa*_vector[GAMMA2]*sinh_2gamma;
}

void ellipses::LogShear::_assignTo(Axes & other) const {
    double gamma = getGamma();
    other[Axes::A] = std::exp(_vector[KAPPA] + gamma);
    other[Axes::B] = std::exp(_vector[KAPPA] - gamma);
    other[Axes::THETA] = 0.5*std::atan2(_vector[GAMMA2],_vector[GAMMA1]);
}

void ellipses::LogShear::_assignTo(Distortion & other) const {
    double gamma = getGamma();
    double e = std::tanh(2*gamma);
    if (gamma < 1E-12) {
        other[Distortion::E1] = other[Distortion::E2] = 0.0;
    } else {
        other[Distortion::E1] = _vector[GAMMA1]*e/gamma;
        other[Distortion::E2] = _vector[GAMMA2]*e/gamma;
    }
    other[Distortion::R] = std::exp(_vector[KAPPA]);
}

ellipses::BaseCore::Jacobian ellipses::LogShear::_dAssignTo(LogShear & other) const {
    other._vector = this->_vector;
    return BaseCore::Jacobian::Identity();
}

ellipses::BaseCore::Jacobian ellipses::LogShear::_dAssignTo(Quadrupole & other) const {
    Distortion tmp;
    BaseCore::Jacobian a = this->_dAssignTo(tmp);
    BaseCore::Jacobian b = static_cast<BaseCore&>(tmp)._dAssignTo(other);
    return b*a;
}

ellipses::BaseCore::Jacobian ellipses::LogShear::_dAssignTo(Axes & other) const {
    BaseCore::Jacobian m;
    double gamma = getGamma();
    double g1 = _vector[GAMMA1];
    double g2 = _vector[GAMMA2];
    other[Axes::A] = std::exp(_vector[KAPPA] + gamma);
    other[Axes::B] = std::exp(_vector[KAPPA] - gamma);
    other[Axes::THETA] = 0.5*std::atan2(g2,g1);
    g1 /= gamma;
    g2 /= gamma;
    m.row(0).setConstant(other[Axes::A]);
    m.row(1).setConstant(other[Axes::B]);
    m(0,0) *= g1;
    m(0,1) *= g2;
    m(1,0) *= -g1;
    m(1,1) *= -g2;
    m(2,0) = -g2/(2.0*gamma);
    m(2,1) = g1/(2.0*gamma);
    m(2,2) = 0.0;
    return m;
}

ellipses::BaseCore::Jacobian ellipses::LogShear::_dAssignTo(Distortion & other) const {
    BaseCore::Jacobian m;
    double gamma = getGamma();
    double e = std::tanh(2*gamma);
    double f1 = 1.0-e*e;
    double f2 = e / gamma;
    double g1 = _vector[GAMMA1]/gamma;
    double g2 = _vector[GAMMA2]/gamma;
    if (gamma < 1E-12) {
        other[Distortion::E1] = other[Distortion::E2] = 0.0;
    } else {
        other[Distortion::E1] = e * g1;
        other[Distortion::E2] = e * g2;
    }
    m(2,2) = other[Distortion::R] = std::exp(_vector[KAPPA]);
    m(0,0) = 2.0*g1*g1*f1 + g2*g2*f2;
    m(1,1) = 2.0*g2*g2*f1 + g1*g1*f2;
    m(0,1) = m(1,0) = g1*g2*(2.0*f1 - f2);
    m(0,2) = m(2,0) = m(1,2) = m(2,1) = 0.0;
    return m;
}
