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
    BaseEllipse(new LogShear(vector.segment<3>(0)), Point2D(vector.segment<2>(2))) 
{ 
    if (doNormalize) normalize(); 
}

void ellipses::LogShear::_assignTo(LogShear & other) const {
    other._vector = this->_vector;
}

void ellipses::LogShear::_assignTo(Quadrupole & other) const {
    double exp_2kappa = std::exp(2.0*_vector[KAPPA]);
    double gamma = getGamma();
    if (gamma < 1E-8) {
        double gamma1 = _vector[GAMMA1];
        double gamma2 = _vector[GAMMA2];
        other[Quadrupole::IXX] = exp_2kappa*(1.0 + 2.0*gamma1 * 4.0*(gamma1*gamma1 + gamma2*gamma2));
        other[Quadrupole::IYY] = exp_2kappa*(1.0 - 2.0*gamma1 * 4.0*(gamma1*gamma1 + gamma2*gamma2));
        other[Quadrupole::IXY] = exp_2kappa*(2.0*gamma2);
    } else {
        double cosh_2gamma = std::cosh(2.0*gamma);
        double sinh_2gamma = std::sinh(2.0*gamma);
        sinh_2gamma /= gamma;
        other[Quadrupole::IXX] = exp_2kappa*(cosh_2gamma + _vector[GAMMA1]*sinh_2gamma);
        other[Quadrupole::IYY] = exp_2kappa*(cosh_2gamma - _vector[GAMMA1]*sinh_2gamma);
        other[Quadrupole::IXY] = exp_2kappa*_vector[GAMMA2]*sinh_2gamma;
    }
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
    double gamma1 = _vector[GAMMA1];
    double gamma2 = _vector[GAMMA2];
    if (gamma < 1E-8) {
        other[Distortion::E1] = 2.0 * gamma1;
        other[Distortion::E2] = 2.0 * gamma2;
    } else {
        other[Distortion::E1] = gamma1*e/gamma;
        other[Distortion::E2] = gamma2*e/gamma;
    }
    other[Distortion::R] = std::exp(_vector[KAPPA]);
}

ellipses::BaseCore::Jacobian ellipses::LogShear::_dAssignTo(LogShear & other) const {
    other._vector = this->_vector;
    return BaseCore::Jacobian::Identity();
}

ellipses::BaseCore::Jacobian ellipses::LogShear::_dAssignTo(Quadrupole & other) const {
    Jacobian result = Jacobian::Zero();
    double exp_2kappa = std::exp(2.0*_vector[KAPPA]);
    double gamma1 = _vector[GAMMA1];
    double gamma2 = _vector[GAMMA2];
    double gamma = getGamma();
    if (gamma < 1E-8) {
        other[Quadrupole::IXX] = exp_2kappa*(1.0 + 2.0*gamma1 * 4.0*(gamma1*gamma1 + gamma2*gamma2));
        other[Quadrupole::IYY] = exp_2kappa*(1.0 - 2.0*gamma1 * 4.0*(gamma1*gamma1 + gamma2*gamma2));
        other[Quadrupole::IXY] = exp_2kappa*(2.0*gamma2);
        result(Quadrupole::IXX, GAMMA1) = exp_2kappa*(2.0 + 8.0*gamma1 + (8.0/3.0)*gamma2*gamma2);
        result(Quadrupole::IYY, GAMMA1) = exp_2kappa*(-2.0 + 8.0*gamma1 - (8.0/3.0)*gamma2*gamma2);
        result(Quadrupole::IXY, GAMMA1) = 0.0;
        result(Quadrupole::IXX, GAMMA2) = exp_2kappa*(8.0*gamma2*(1.0 + (2.0/3.0)*gamma1));
        result(Quadrupole::IYY, GAMMA2) = exp_2kappa*(8.0*gamma2*(1.0 - (2.0/3.0)*gamma1));
        result(Quadrupole::IXY, GAMMA2) = exp_2kappa*(2.0 + 24.0*gamma2);
    } else {
        double cosh_2gamma = std::cosh(2.0*gamma);
        double sinh_2gamma = std::sinh(2.0*gamma);
        sinh_2gamma /= gamma;
        other[Quadrupole::IXX] = exp_2kappa*(cosh_2gamma + gamma1*sinh_2gamma);
        other[Quadrupole::IYY] = exp_2kappa*(cosh_2gamma - gamma1*sinh_2gamma);
        other[Quadrupole::IXY] = exp_2kappa*gamma2*sinh_2gamma;
        double csh = exp_2kappa * cosh_2gamma * 2.0 / (gamma*gamma);
        double snh = exp_2kappa * sinh_2gamma / (gamma*gamma);
        result(Quadrupole::IXX, GAMMA1) = csh*gamma1*gamma1 + snh*(2*gamma*gamma*gamma1 + gamma2*gamma2);
        result(Quadrupole::IXX, GAMMA2) = csh*gamma1*gamma2 + snh*(2*gamma*gamma*gamma2 - gamma1*gamma2);
        result(Quadrupole::IYY, GAMMA1) = -csh*gamma1*gamma1 + snh*(2*gamma*gamma*gamma1 - gamma2*gamma2);
        result(Quadrupole::IYY, GAMMA2) = -csh*gamma1*gamma2 + snh*(2*gamma*gamma*gamma2 + gamma1*gamma2);
        result(Quadrupole::IXY, GAMMA1) = (csh - snh)*gamma1*gamma2;
        result(Quadrupole::IXY, GAMMA2) = snh*gamma1*gamma1 + csh*gamma2*gamma2;
    }
    result(Quadrupole::IXX, KAPPA) = other[Quadrupole::IXX] * 2.0;
    result(Quadrupole::IYY, KAPPA) = other[Quadrupole::IYY] * 2.0;
    result(Quadrupole::IXY, KAPPA) = other[Quadrupole::IXY] * 2.0;
    return result;
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
    Jacobian result = Jacobian::Zero();
    double gamma = getGamma();
    double e = std::tanh(2*gamma);
    double gamma1 = _vector[GAMMA1];
    double gamma2 = _vector[GAMMA2];
    if (gamma < 1E-8) {
        other[Distortion::E1] = 2.0 * gamma1;
        other[Distortion::E2] = 2.0 * gamma2;
        result(Distortion::E1, GAMMA1) = 2.0 - 48.0*gamma1*gamma1;
        result(Distortion::E2, GAMMA2) = 2.0 - 48.0*gamma2*gamma2;
    } else {
        other[Distortion::E1] = gamma1 * e / gamma;
        other[Distortion::E2] = gamma2 * e / gamma;
        double cth = gamma1 / gamma;
        double sth = gamma2 / gamma;
        double tmp1 = e + 2*gamma*(e + 1.0)*(e - 1.0);
        result(Distortion::E1, GAMMA1) = (e - cth*cth*tmp1) / gamma;
        result(Distortion::E2, GAMMA1) = result(Distortion::E1, GAMMA2) = -cth*sth*tmp1/gamma;
        result(Distortion::E2, GAMMA2) = (e - sth*sth*tmp1) / gamma;
    }
    result(Distortion::R, KAPPA) = other[Distortion::R] = std::exp(_vector[KAPPA]);
    return result;
}
