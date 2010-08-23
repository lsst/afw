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

ellipses::DistortionEllipse::DistortionEllipse(ParameterVector const & vector, bool doNormalize) :
    BaseEllipse(new Distortion(vector.segment<3>(0)), PointD(vector.segment<2>(2))) 
{ 
    if (doNormalize) normalize(); 
}

void ellipses::Distortion::_assignTo(Distortion & other) const {
    other._vector = this->_vector;
}

void ellipses::Distortion::_assignTo(Quadrupole & other) const {
    double e = getE();
    e *= e;
    double v = _vector[R] * _vector[R] / std::sqrt(1.0-e);
    other[Quadrupole::IXX] = v*(1.0+_vector[E1]);
    other[Quadrupole::IYY] = v*(1.0-_vector[E1]);
    other[Quadrupole::IXY] = v*_vector[E2];
}

void ellipses::Distortion::_assignTo(Axes & other) const {
    double e = getE();
    double q = std::pow((1.0-e)/(1.0+e),0.25);
    other[Axes::A] = _vector[R]/q;
    other[Axes::B] = _vector[R]*q;
    other[Axes::THETA] = 0.5*std::atan2(_vector[E2],_vector[E1]);
}

void ellipses::Distortion::_assignTo(LogShear & other) const {
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

ellipses::BaseCore::Jacobian ellipses::Distortion::_dAssignTo(Distortion & other) const {
    other._vector = this->_vector;
    return BaseCore::Jacobian::Identity();
}

ellipses::BaseCore::Jacobian ellipses::Distortion::_dAssignTo(Quadrupole & other) const {
    BaseCore::Jacobian m;
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

ellipses::BaseCore::Jacobian ellipses::Distortion::_dAssignTo(Axes & other) const {
    BaseCore::Jacobian m;
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
    m.corner<2,2>(Eigen::TopLeft).setConstant(_vector[R] / (2 * e * std::pow(dp*dm,3)));
    dm *= dm;
    dp *= dp;
    m(0,0) *= _vector[E1] / dm;
    m(0,1) *= _vector[E2] / dm;
    m(1,0) *= -_vector[E1] / dp;
    m(1,1) *= -_vector[E2] / dp;
    return m;
}

ellipses::BaseCore::Jacobian ellipses::Distortion::_dAssignTo(LogShear & other) const {
    BaseCore::Jacobian m;
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
