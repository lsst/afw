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

namespace lsst { namespace afw { namespace geom { namespace ellipses {

BaseCore::Registrar<Axes> Axes::registrar;

std::string Axes::getName() const { return "Axes"; }

void Axes::normalize() {
    if (_vector[A] < 0 || _vector[B] < 0)
        throw LSST_EXCEPT(lsst::pex::exceptions::InvalidParameterError,
                          "Major and minor axes cannot be negative.");
    if (_vector[A] < _vector[B]) {
        std::swap(_vector[A], _vector[B]);
        _vector[THETA] += M_PI_2;
    }
    if (_vector[THETA] > M_PI_2 || _vector[THETA] <= -M_PI_2) {
        _vector[THETA] -= M_PI * std::ceil(_vector[THETA] / M_PI - 0.5);
    }
}

void Axes::readParameters(double const * iter) {
    setA(*iter++);
    setB(*iter++);
    setTheta(*iter++);
}

void Axes::writeParameters(double * iter) const {
    *iter++ = getA();
    *iter++ = getB();
    *iter++ = getTheta();
}

void Axes::_assignToQuadrupole(double & ixx, double & iyy, double & ixy) const {
    BaseCore::_assignAxesToQuadrupole(_vector[A], _vector[B], _vector[THETA], ixx, iyy, ixy);
}

BaseCore::Jacobian Axes::_dAssignToQuadrupole(double & ixx, double & iyy, double & ixy) const {
    return BaseCore::_dAssignAxesToQuadrupole(_vector[A], _vector[B], _vector[THETA], ixx, iyy, ixy);
}

void Axes::_assignToAxes(double & a, double & b, double & theta) const {
    a = _vector[A];
    b = _vector[B];
    theta = _vector[THETA];
}

BaseCore::Jacobian Axes::_dAssignToAxes(double & a, double & b, double & theta) const {
    a = _vector[A];
    b = _vector[B];
    theta = _vector[THETA];
    return Jacobian::Identity();
}

void Axes::_assignFromQuadrupole(double ixx, double iyy, double ixy) {
    BaseCore::_assignQuadrupoleToAxes(ixx, iyy, ixy, _vector[A], _vector[B], _vector[THETA]);
    normalize();
}

BaseCore::Jacobian Axes::_dAssignFromQuadrupole(double ixx, double iyy, double ixy) {
    return BaseCore::_dAssignQuadrupoleToAxes(ixx, iyy, ixy, _vector[A], _vector[B], _vector[THETA]);
    normalize();
}

void Axes::_assignFromAxes(double a, double b, double theta) {
    _vector[A] = a;
    _vector[B] = b;
    _vector[THETA] = theta;
}

BaseCore::Jacobian Axes::_dAssignFromAxes(double a, double b, double theta) {
    _vector[A] = a;
    _vector[B] = b;
    _vector[THETA] = theta;
    return Jacobian::Identity();
}

}}}} // namespace lsst::afw::geom::ellipses
